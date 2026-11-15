"""De-limiting with the De-limiter network (Jeon & Lee, "Music De-limiter Networks via Sample-wise Gain
Inversion", WASPAA 2023, https://github.com/jeonchangbin49/De-limiter).

The model predicts, per sample, the gain a mastering limiter took away. We only use its gain curve: it is
applied to every stem (the limiter acted on the whole mix) with a chosen modulation depth. The model code and
weights stay in their own checkout (AUDIOFIX_DELIMITER, default ~/.cache/audiofix/De-limiter)."""
import json, os, sys
import numpy as np

from . import SR

REPO = os.environ.get("AUDIOFIX_DELIMITER", os.path.expanduser("~/.cache/audiofix/De-limiter"))


def available():
    if not os.path.exists(os.path.join(REPO, "weight", "all.pth")):
        return False
    try:
        import torch  # noqa: F401
        import asteroid_filterbanks  # noqa: F401
        from asteroid.masknn import TDConvNet  # noqa: F401
        import dotmap, einops  # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def _load():
    import torch
    from dotmap import DotMap
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from models import load_model_with_args
    a = DotMap(json.load(open(os.path.join(REPO, "weight", "all.json")))["args"])
    m = load_model_with_args(a)
    m.load_state_dict(torch.load(os.path.join(REPO, "weight", "all.pth"), map_location="cpu", weights_only=True))
    return m.eval()


def gain_curve(x, chunk=20.0, overlap=2.0):
    """Sample-wise gain curve (0..1) for a 44.1 kHz stereo signal, inferred in overlapping chunks on CPU."""
    import torch
    model = _load()
    T = len(x); C = int(chunk * SR); O = int(overlap * SR)
    acc = np.zeros(T); ws = np.zeros(T)
    s = 0
    while True:
        e = min(T, s + C)
        seg = torch.from_numpy(np.ascontiguousarray(x[s:e].T, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            g, _ = model(seg)
        g = g[0, 0].numpy().astype(np.float64)
        w = np.ones(e - s)
        if s > 0:
            w[:O] = np.linspace(0, 1, O)
        if e < T:
            w[-O:] = np.linspace(1, 0, O)
        acc[s:e] += g * w; ws[s:e] += w
        if e == T:
            break
        s = e - O
    return (acc / np.maximum(ws, 1e-9)).astype(np.float32)


def modulation(g, depth):
    """Gain curve -> applied gain: its swings around the median, scaled by depth (2 = twice as deep)."""
    gd = 20 * np.log10(np.maximum(g.astype(np.float64), 1e-6))
    gd -= np.median(gd)
    return (10 ** (depth * gd / 20)).astype(np.float32)
