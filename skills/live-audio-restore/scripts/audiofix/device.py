"""Compute device for the neural models (Apollo, demucs): CUDA on an NVIDIA GPU, MPS on Apple silicon, else CPU.

AUDIOFIX_DEVICE=auto|cuda|mps|cpu forces one; a device that is not available falls back to CPU with a message.
Callers also fall back to CPU when a GPU run fails (out of memory, an operation MPS does not support), so a GPU
can only make a run faster, never break it."""
import os


def _torch():
    try:
        import torch
        return torch
    except Exception:                                            # noqa: BLE001
        return None


def available():
    """{'cuda': bool, 'mps': bool}"""
    torch = _torch()
    if torch is None:
        return {"cuda": False, "mps": False}
    mps = getattr(torch.backends, "mps", None)
    return {"cuda": bool(torch.cuda.is_available()), "mps": bool(mps is not None and mps.is_available())}


def pick(log=None):
    want = os.environ.get("AUDIOFIX_DEVICE", "auto").strip().lower()
    av = available()
    if want in ("", "auto"):
        return "cuda" if av["cuda"] else "mps" if av["mps"] else "cpu"
    if want in ("cuda", "gpu") and av["cuda"]:
        return "cuda"
    if want == "mps" and av["mps"]:
        return "mps"
    if want != "cpu" and log:
        log(f"[device] AUDIOFIX_DEVICE={want} is not available here: using the CPU")
    return "cpu"


def describe():
    """Human-readable line for `check` and the logs."""
    torch = _torch()
    d = pick()
    if torch is None:
        return "cpu (PyTorch not installed)"
    if d == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:                                        # noqa: BLE001 (driver trouble: the run will fall back)
            return "cuda"
    if d == "mps":
        return "mps (Apple silicon GPU)"
    return f"cpu ({os.cpu_count()} threads; no CUDA or MPS GPU found)"
