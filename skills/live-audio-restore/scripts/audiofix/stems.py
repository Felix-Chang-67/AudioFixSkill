"""Source separation with demucs (htdemucs: vocals / drums / bass / other), cached in the work directory."""
import os, subprocess, sys
import numpy as np

STEMS = ("vocals", "drums", "bass", "other")


def available():
    try:
        import demucs  # noqa: F401
        import torch   # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def separate(wav_path, work, model="htdemucs"):
    """Run demucs once; returns the folder holding vocals/drums/bass/other.wav."""
    name = os.path.splitext(os.path.basename(wav_path))[0]
    out = os.path.join(work, "stems")
    folder = os.path.join(out, model, name)
    if all(os.path.exists(os.path.join(folder, f"{s}.wav")) for s in STEMS):
        return folder
    if os.path.islink(folder) and not os.path.exists(folder):     # a cache link whose target is gone
        os.unlink(folder)
    from . import device
    dev = device.pick()
    cmd = [sys.executable, "-m", "demucs", "-n", model, "--float32", "-o", out, "-d", dev, wav_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if dev == "cpu":
            raise
        print(f"[stems] demucs on {dev} failed: running on the CPU instead", flush=True)
        subprocess.run(cmd[:-2] + ["cpu", wav_path], check=True)
    return folder


def load(folder, n):
    import soundfile as sf
    st = {}
    for s in STEMS:
        y, _ = sf.read(os.path.join(folder, f"{s}.wav"), dtype="float32", always_2d=True)
        y = y[:, :2] if y.shape[1] >= 2 else np.repeat(y, 2, axis=1)
        st[s] = y[:n] if len(y) >= n else np.vstack([y, np.zeros((n - len(y), 2), np.float32)])
    return st
