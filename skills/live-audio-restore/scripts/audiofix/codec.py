"""Codec repair with Apollo (Li et al., JusperLee/Apollo, CC BY-SA 4.0; cloned by setup.sh, not part of this repo).

A low-bitrate AAC/MP3 leaves the band below its cut-off sparse and peaky: whole time-frequency tiles are zeroed
("birdies", holes). Boosting that band, or shifting it up to rebuild the top, makes the highs sound fake and harsh.
Apollo is a network trained to restore lossy-coded music; it refills that texture. On an audience recording the
8-12.5 kHz band went from a spectral flatness of 0.08 to 0.26 (the studio master of the same song: 0.42), with the
tone otherwise unchanged. Its own extension above the cut-off is weak, so the band extension still runs after it.

The checkpoint is loaded with torch's restricted unpickler and an allowlist of the plain classes it references
(collections, omegaconf containers, torch storages); the omegaconf classes are stand-ins when omegaconf 2 is not
installed, so no arbitrary code from the file can run. CPU speed: about 6x real time (a 5 min song ~ 30 min)."""
import os, sys, time, types
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


def repair(x, chunk_s=10.0, overlap_s=1.0, log=print):
    """x: (n, 2) float32 at 44.1 kHz -> repaired (n, 2) float32. Chunked, crossfaded."""
    import torch
    model = _load_model()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t = torch.from_numpy(np.ascontiguousarray(x.T, dtype=np.float32)).unsqueeze(0)
    n = t.shape[-1]
    C, O = int(chunk_s * SR), int(overlap_s * SR)
    out = torch.zeros_like(t)
    wsum = torch.zeros(1, 1, n)
    t0, start = time.time(), 0
    with torch.inference_mode():
        while True:
            end = min(n, start + C)
            seg = t[..., start:end]
            if seg.shape[-1] < C:
                seg = torch.nn.functional.pad(seg, (0, C - seg.shape[-1]))
            o = model(seg)[..., :end - start]
            w = torch.ones(end - start)
            if start > 0:
                w[:O] = torch.linspace(0, 1, O)
            if end < n:
                w[-O:] = torch.linspace(1, 0, O)
            out[..., start:end] += o * w
            wsum[..., start:end] += w
            if start == 0 or (start // C) % 6 == 0:
                log(f"[codec] Apollo {end / SR:.0f} / {n / SR:.0f} s ({time.time() - t0:.0f} s)")
            if end == n:
                break
            start = end - O
    y = (out / wsum.clamp_min(1e-6)).squeeze(0).numpy().T
    return np.ascontiguousarray(y, dtype=np.float32)
