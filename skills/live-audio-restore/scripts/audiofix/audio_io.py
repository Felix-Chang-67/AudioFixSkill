"""Reading, writing and encoding audio. Works with ffmpeg when present, falls back to WAV-only I/O."""
import json, shutil, subprocess
import numpy as np

from . import SR


def have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path):
    """Container/codec facts of the source (best effort)."""
    info = {"path": str(path)}
    if have_ffmpeg():
        r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                           capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            a = next(s for s in d.get("streams", []) if s.get("codec_type") == "audio")
            info.update(codec=a.get("codec_name"), profile=a.get("profile"), sample_rate=int(a.get("sample_rate", 0)),
                        channels=a.get("channels"), bit_rate=int(a.get("bit_rate") or d["format"].get("bit_rate") or 0),
                        duration=float(d["format"].get("duration", 0)),
                        tags=d["format"].get("tags", {}))
        except Exception as e:                                   # noqa: BLE001
            info["probe_error"] = str(e)
    else:
        try:
            import soundfile as sf
            i = sf.info(str(path))
            info.update(codec=i.subtype, sample_rate=i.samplerate, channels=i.channels, duration=i.duration)
        except Exception as e:                                   # noqa: BLE001
            info["probe_error"] = str(e)
    return info


def load(path, sr=SR):
    """Any format -> float32 array (n, 2) at sr. Mono sources are duplicated to two identical channels."""
    if have_ffmpeg():
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(sr),
                              "-f", "f32le", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    try:
        import soundfile as sf
        y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    except ImportError:
        from scipy.io import wavfile
        fs, y = wavfile.read(str(path))
        y = y.astype(np.float32) / (np.iinfo(y.dtype).max if y.dtype.kind == "i" else 1.0)
        y = y.reshape(len(y), -1)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = y[:, :2]
    if fs != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(fs, sr)
        y = resample_poly(y, sr // g, fs // g, axis=0).astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


def save_wav(path, y, sr=SR, bits=24):
    y = np.clip(np.asarray(y, np.float32), -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(path), y, sr, subtype={16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[bits])
    except ImportError:
        from scipy.io import wavfile
        if bits == 32:
            wavfile.write(str(path), sr, y)
        else:
            wavfile.write(str(path), sr, (y * 32767).astype(np.int16))


def encode_aac(wav_path, out_path, kbps=256):
    if not have_ffmpeg():
        return None
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav_path), "-c:a", "aac", "-b:a", f"{kbps}k",
                    "-movflags", "+faststart", str(out_path)], check=True)
    return out_path
