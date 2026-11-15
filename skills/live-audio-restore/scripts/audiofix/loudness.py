"""ITU-R BS.1770 loudness, EBU Tech 3342 loudness range and true peak, self-contained (numpy/scipy only)."""
import numpy as np
from scipy.signal import lfilter

from . import SR
from .dsp import db, isp_peaks


def _biquads(fs):
    """K-weighting: high shelf (+4 dB above ~1.5 kHz) and high pass (~38 Hz), RBJ form as in pyloudnorm."""
    G, Q, fc = 4.0, 1 / np.sqrt(2), 1500.0
    A = 10 ** (G / 40); w0 = 2 * np.pi * fc / fs; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    hs = ([A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
           A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)],
          [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
           (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al])
    Q, fc = 0.5, 38.0
    w0 = 2 * np.pi * fc / fs; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    hp = ([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al])
    return hs, hp


def kweight(y, fs=SR):
    (b1, a1), (b2, a2) = _biquads(fs)
    z = lfilter(b1, a1, y, axis=0)
    return lfilter(b2, a2, z, axis=0)


def _block_power(k, win, hop):
    p = (k.astype(np.float64) ** 2).sum(axis=1)                 # channel weights 1.0 for L and R
    c = np.concatenate([[0.0], np.cumsum(p)])
    starts = np.arange(0, max(1, len(p) - win + 1), hop)
    return (c[np.minimum(starts + win, len(p))] - c[starts]) / win


def integrated(y, fs=SR):
    """Gated integrated loudness, LUFS."""
    if len(y) < int(0.4 * fs):
        return -70.0
    z = _block_power(kweight(y, fs), int(0.4 * fs), int(0.1 * fs))
    lk = -0.691 + 10 * np.log10(z + 1e-30)
    z1 = z[lk > -70]
    if not len(z1):
        return -70.0
    rel = -0.691 + 10 * np.log10(z1.mean()) - 10
    z2 = z1[-0.691 + 10 * np.log10(z1) > rel]
    return float(-0.691 + 10 * np.log10(z2.mean())) if len(z2) else -70.0


def short_term(y, fs=SR, hop_s=0.1):
    z = _block_power(kweight(y, fs), int(3.0 * fs), int(hop_s * fs))
    return -0.691 + 10 * np.log10(z + 1e-30)


def loudness_range(y, fs=SR):
    """EBU Tech 3342 LRA: 10th to 95th percentile of gated short-term loudness."""
    st = short_term(y, fs)
    st = st[st > -70]
    if len(st) < 2:
        return 0.0
    pw = 10 ** ((st + 0.691) / 10)
    rel = -0.691 + 10 * np.log10(pw.mean()) - 20
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) > 1 else 0.0


def true_peak(y):
    return float(db(isp_peaks(y).max()))


def crest_3s(y, fs=SR):
    """Median over 3 s windows of peak / RMS in dB: how much the hits stand out locally."""
    w = 3 * fs
    out = []
    for i in range(0, len(y) - w + 1, w):
        seg = y[i:i + w]
        r = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
        if r > 1e-5:
            out.append(db(np.abs(seg).max() / r))
    return float(np.median(out)) if out else 0.0


def summary(y, fs=SR):
    I = integrated(y, fs)
    tp = true_peak(y)
    return {"lufs": round(I, 2), "true_peak_dbtp": round(tp, 2), "plr_db": round(tp - I, 2),
            "lra_lu": round(loudness_range(y, fs), 2), "crest3s_db": round(crest_3s(y, fs), 2)}
