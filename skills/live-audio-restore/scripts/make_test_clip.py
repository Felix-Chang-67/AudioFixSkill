#!/usr/bin/env python3
"""Make synthetic, copyright-free degraded live clips with known problems (for tests and demos).

  make_test_clip.py song out.m4a   ~40 s band: drums, bass, two guitar layers (one in anti-phase), a sung-vowel lead,
                                   crowd; then a broadcast chain: heavy limiting, AAC 128 kb/s (15-16 kHz wall)
  make_test_clip.py mc out.m4a     ~40 s talk (espeak-ng) through a PA: band-limited, reverberant, low level, with
                                   crowd cheers between sentences that are louder than the talk

Known problems are printed so a test can check the diagnosis found them.
"""
import os, subprocess, sys, tempfile
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
rng = np.random.default_rng(7)


def bp(x, lo, hi, order=4):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def lp(x, fc, order=4):
    return sosfilt(butter(order, fc, btype="low", fs=SR, output="sos"), x)


def saw(f, n):
    t = np.arange(n) / SR
    return 2 * ((t * f) % 1.0) - 1


def crowd(n, level=0.05):
    c = bp(rng.normal(0, 1, (n, 2)).T, 300, 6000).T
    env = np.convolve(np.abs(rng.normal(0, 1, n)), np.ones(SR // 2) / (SR // 2), "same")
    return c * (env / env.max())[:, None] * level


def reverb(x, rt=1.2, wet=0.25):
    n = int(rt * SR)
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-6.9 * np.arange(n) / n)[:, None]
    ir = lp(ir.T, 5000).T
    out = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return x + out / np.abs(out).max() * np.abs(x).max() * wet


def broadcast_limit(y, gain_db=10.0, ceiling=0.89):
    """Crude broadcast processor: lots of gain into a fast hold-and-smooth limiter (flattens everything)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    y = y * 10 ** (gain_db / 20)
    g = np.minimum(1.0, ceiling / np.maximum(np.abs(y).max(axis=1), 1e-9))
    k = int(0.003 * SR)
    g = uniform_filter1d(minimum_filter1d(g, size=3 * k), size=k)
    return np.clip(y * g[:, None], -0.95, 0.95)


def song(dur=40.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    beat = 60 / 140
    mix = np.zeros((n, 2))
    # drums: kick on beats, snare on 2 and 4, hats on 8ths
    for k, tb in enumerate(np.arange(0, dur, beat)):
        i = int(tb * SR)
        m = min(n - i, int(0.35 * SR))
        tt = np.arange(m) / SR
        mix[i:i + m] += (np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9))[:, None] * 0.9
        if k % 2:
            s = bp(rng.normal(0, 1, m), 180, 7000) * np.exp(-tt * 18) * 0.6
            mix[i:i + m] += s[:, None]
        for off in (0, beat / 2):
            j = int((tb + off) * SR); mm = min(n - j, int(0.05 * SR))
            if mm > 0:
                hh = bp(rng.normal(0, 1, mm), 7000, 16000) * np.exp(-np.arange(mm) / SR * 80) * 0.25
                mix[j:j + mm] += np.stack([hh, hh * 0.8], 1)
    roots = [82.4, 110.0, 98.0, 73.4]                                # E A G D
    bar = 4 * beat
    for k, tb in enumerate(np.arange(0, dur, bar)):
        i, j = int(tb * SR), min(n, int((tb + bar) * SR))
        f = roots[k % 4]
        bass = lp(saw(f, j - i), 600) * 0.35
        mix[i:j] += bass[:, None]
        chord = sum(saw(f * 2 * r, j - i) for r in (1, 1.26, 1.5))
        g1 = lp(np.tanh(3 * chord), 4500) * 0.12                     # layer 1: normal stereo
        g2 = lp(np.tanh(3 * sum(saw(f * 4 * r * 1.003, j - i) for r in (1, 1.5))), 5000) * 0.14
        mix[i:j, 0] += g1 * 1.0 + g2                                 # layer 2 in anti-phase between L and R
        mix[i:j, 1] += g1 * 0.8 - g2
    # lead "voice": sung vowels with vibrato and formants
    notes = [330, 392, 440, 392, 330, 294, 330, 392]
    for k, tb in enumerate(np.arange(0, dur, beat * 2)):
        i, j = int(tb * SR), min(n, int((tb + beat * 1.8) * SR))
        tt = np.arange(j - i) / SR
        f0 = notes[k % len(notes)] * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * tt))
        ph = 2 * np.pi * np.cumsum(f0) / SR
        src = sum(np.sin(h * ph) / h for h in range(1, 30))
        v = sum(bp(src, fc - 80, fc + 80, 2) * a for fc, a in ((700, 1.0), (1200, 0.6), (2600, 0.35), (3400, 0.25)))
        v *= np.minimum(1, tt / 0.05) * np.minimum(1, (tt[-1] - tt) / 0.1) * 0.5
        mix[i:j] += np.stack([v, v], 1)
    mix += crowd(n, 0.03)
    y = broadcast_limit(mix / np.abs(mix).max() * 0.5, gain_db=12)
    return y, ["heavily limited (very low loudness range)", "guitar layer in anti-phase between L and R",
               "AAC 128 kb/s band limit (a wall at about 16-18 kHz, depending on the encoder)"]


def mc(dur=40.0):
    lines = ["Thank you so much, everybody. It is really hot in here tonight.",
             "Before the show, our guitarist stepped on his pedal and it broke in two.",
             "Then the drum beater snapped during the rehearsal.",
             "So please take care of yourselves, and let's enjoy the rest of the night together."]
    tmp = tempfile.mkdtemp()
    n = int(dur * SR)
    voice = np.zeros(n)
    t = 1.5
    for k, s in enumerate(lines):
        p = os.path.join(tmp, f"{k}.wav")
        subprocess.run(["espeak-ng", "-v", "en", "-s", "150", "-w", p, s], check=True)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        v = np.frombuffer(raw, np.float32).astype(np.float64)
        i = int(t * SR); m = min(len(v), n - i)
        voice[i:i + m] += v[:m]
        t += m / SR + 4.0                                               # 4 s gap for the crowd
    voice = bp(voice, 180, 5500, 2)                                     # PA and an audience-position recorder
    y = reverb(np.stack([voice, voice * 0.95], 1), rt=1.4, wet=0.35) * 0.3
    # cheers in the gaps, louder than the talk
    act = np.convolve(np.abs(voice) > 0.01, np.ones(SR // 2), "same") > 0
    cheer = crowd(n, 1.0)
    gap_env = np.convolve((~act).astype(float), np.ones(SR // 3) / (SR // 3), "same")
    y += cheer * gap_env[:, None] * 0.25 + crowd(n, 0.02)
    y += lp(rng.normal(0, 1, (n, 2)).T, 120).T * 0.02                   # room rumble
    y *= 10 ** (-22 / 20) / max(np.abs(y).max(), 1e-9)                   # recorded far too low
    return y, ["crowd cheers between sentences louder than the talk", "PA-band-limited, reverberant voice",
               "very low recording level"]


def main():
    kind, out = sys.argv[1], sys.argv[2]
    y, problems = song() if kind == "song" else mc()
    tmp = out + ".tmp.wav"
    import soundfile as sf
    sf.write(tmp, y.astype(np.float32), SR, subtype="FLOAT")
    if out.endswith(".wav"):
        os.replace(tmp, out)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:a", "aac", "-b:a", "128k", out], check=True)
        os.remove(tmp)
    print(f"{out}: {kind}, known problems: " + "; ".join(problems))


if __name__ == "__main__":
    main()
