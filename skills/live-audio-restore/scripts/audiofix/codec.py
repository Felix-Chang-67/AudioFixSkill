"""Codec repair with Apollo (Li et al., JusperLee/Apollo, CC BY-SA 4.0; cloned by setup.sh, not part of this repo).

A low-bitrate AAC/MP3 leaves the band below its cut-off sparse and peaky: whole time-frequency tiles are zeroed
("birdies", holes). Boosting that band, or shifting it up to rebuild the top, makes the highs sound fake and harsh.
Apollo is a network trained to restore lossy-coded music; it refills that texture. On an audience recording the
8-12.5 kHz band went from a spectral flatness of 0.08 to 0.26 (the studio master of the same song: 0.42), with the
tone otherwise unchanged. Its own extension above the cut-off is weak, so the band extension still runs after it.

The checkpoint is loaded with torch's restricted unpickler and an allowlist of the plain classes it references
(collections, omegaconf containers, torch storages); the omegaconf classes are stand-ins when omegaconf 2 is not
installed, so no arbitrary code from the file can run.

Speed: on the CPU about 6x real time (a 5 min song takes about 30 min). It runs on an NVIDIA (CUDA) or Apple
silicon (MPS) GPU when there is one (device.py), and falls back to the CPU if the GPU run fails.

Resumable: with a checkpoint folder, every finished 10 s chunk is saved there, and a run that was cut off (an
assistant's command time-out, a closed laptop) continues from the last saved chunk when the same command is run
again. Assistants whose tools stop commands after a few minutes used to restart the 30 min job from zero each time."""
import hashlib, json, os, shutil, sys, time, types
import numpy as np

from . import SR

REPO = os.environ.get("AUDIOFIX_APOLLO", os.path.expanduser("~/.cache/audiofix/Apollo"))
CKPT_REPO, CKPT_FILE = "JusperLee/Apollo", "pytorch_model.bin"


def available():
    if not os.path.isfile(os.path.join(REPO, "look2hear", "models", "apollo.py")):
        return False
    try:
        import torch  # noqa: F401
        import huggingface_hub  # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


class _Stub:
    """Stand-in for an omegaconf container: keeps whatever state the pickle gives it, runs nothing."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


def _stub_omegaconf():
    """Register stand-in omegaconf 2 classes under their module names (only if the real ones are not importable)."""
    try:
        from omegaconf.base import ContainerMetadata  # noqa: F401  (omegaconf >= 2 present: use it)
        import omegaconf
        from omegaconf.base import Metadata
        from omegaconf.nodes import AnyNode
        return [omegaconf.DictConfig, omegaconf.ListConfig, ContainerMetadata, Metadata, AnyNode], []
    except Exception:                                            # noqa: BLE001
        pass
    made, names = [], []
    for mod, cls in (("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
                     ("omegaconf.base", "ContainerMetadata"), ("omegaconf.base", "Metadata"),
                     ("omegaconf.nodes", "AnyNode")):
        m = sys.modules.get(mod)
        if m is None or not hasattr(m, "__audiofix_stub__"):
            m = types.ModuleType(mod)
            m.__audiofix_stub__ = True
            sys.modules[mod] = m
            names.append(mod)
        c = type(cls, (_Stub,), {"__module__": mod})
        setattr(m, cls, c)
        made.append(c)
    return made, names


def _load_model():
    import collections, typing
    import torch
    from huggingface_hub import hf_hub_download
    from torch.torch_version import TorchVersion
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import look2hear.models
    path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE)
    oc, stubbed = _stub_omegaconf()
    allow = [collections.OrderedDict, collections.defaultdict, typing.Any, TorchVersion,
             dict, list, tuple, set, str, int, float, bool] + oc
    try:
        with torch.serialization.safe_globals(allow):
            conf = torch.load(path, map_location="cpu", weights_only=True)
    finally:
        for mod in stubbed:
            sys.modules.pop(mod, None)
    model = look2hear.models.get(conf["model_name"])(sr=SR, win=20, feature_dim=256, layer=6)
    model.load_state_dict(conf["state_dict"])
    return model.eval()


def repair(x, chunk_s=10.0, overlap_s=1.0, log=print, checkpoint=None):
    """x: (n, 2) float32 at 44.1 kHz -> repaired (n, 2) float32. Chunked, crossfaded. Runs on the GPU when there is
    one (CUDA or MPS, see device.py) and falls back to the CPU if the GPU run fails. checkpoint: a folder where each
    finished chunk is kept, so that a run cut off part way resumes there (delete it once the result is saved)."""
    from . import device
    dev = device.pick(log)
    log(f"[codec] Apollo on {device.describe() if dev != 'cpu' else 'cpu'}")
    ck = _checkpoint(checkpoint, x, chunk_s, overlap_s, log)
    if dev != "cpu":
        try:
            return _run(x, chunk_s, overlap_s, log, dev, ck)
        except Exception as e:                                   # noqa: BLE001 (out of memory, unsupported op ...)
            log(f"[codec] {dev} run failed ({str(e)[:120]}): running on the CPU instead")
    return _run(x, chunk_s, overlap_s, log, "cpu", ck)


def _checkpoint(folder, x, chunk_s, overlap_s, log):
    """The chunk folder for this very input and chunking; chunks saved for anything else are dropped."""
    if not folder:
        return None
    key = {"sha1": hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest(),
           "n": int(len(x)), "chunk_s": chunk_s, "overlap_s": overlap_s}
    meta = os.path.join(folder, "meta.json")
    try:
        with open(meta) as fh:
            same = json.load(fh) == key
    except (OSError, ValueError):
        same = False
    if not same and os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    if not same:
        with open(meta, "w") as fh:
            json.dump(key, fh)
    done = len([f for f in os.listdir(folder) if f.startswith("chunk_")])
    if done:
        log(f"[codec] resuming: {done} chunk(s) of {chunk_s:.0f} s already done in {folder}")
    return folder


def _run(x, chunk_s, overlap_s, log, dev, ck=None):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x.T, dtype=np.float32)).unsqueeze(0)
    n = t.shape[-1]
    C, O = int(chunk_s * SR), int(overlap_s * SR)
    out = torch.zeros_like(t)
    wsum = torch.zeros(1, 1, n)
    t0, start, i, model, ran = time.time(), 0, 0, None, 0
    total = 1 + max(0, -(-(n - C) // (C - O)))
    with torch.inference_mode():
        while True:
            end = min(n, start + C)
            path = os.path.join(ck, f"chunk_{i:04d}.npy") if ck else None
            if path and os.path.exists(path):
                o = torch.from_numpy(np.load(path))
            else:
                if model is None:
                    model = _load_model().to(dev)
                    if dev == "cpu":
                        torch.set_num_threads(max(1, os.cpu_count() or 1))
                seg = t[..., start:end]
                if seg.shape[-1] < C:
                    seg = torch.nn.functional.pad(seg, (0, C - seg.shape[-1]))
                o = model(seg.to(dev))[..., :end - start].to("cpu")
                ran += 1
                if path:
                    np.save(path + ".tmp.npy", o.numpy())
                    os.replace(path + ".tmp.npy", path)
                el = time.time() - t0
                log(f"[codec] Apollo {end / SR:.0f} / {n / SR:.0f} s, chunk {i + 1} of {total} "
                    f"({el:.0f} s; about {el / ran * (total - i - 1) / 60:.0f} min left)")
            w = torch.ones(end - start)
            if start > 0:
                w[:O] = torch.linspace(0, 1, O)
            if end < n:
                w[-O:] = torch.linspace(1, 0, O)
            out[..., start:end] += o * w
            wsum[..., start:end] += w
            if end == n:
                break
            start, i = end - O, i + 1
    y = (out / wsum.clamp_min(1e-6)).squeeze(0).numpy().T
    return np.ascontiguousarray(y, dtype=np.float32)
