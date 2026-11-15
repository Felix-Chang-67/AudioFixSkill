"""DSP building blocks shared by the music and speech paths.

Everything works on float32 arrays shaped (n, 2) at 44.1 kHz. numba speeds up the envelope followers and the
limiter release; without it the same functions run as vectorised approximations.
"""
import os

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter, uniform_filter1d
from scipy.signal import butter, firwin2, istft, lfilter, oaconvolve, resample_poly, sosfiltfilt, stft, welch

from . import SR

try:
    if os.environ.get("AUDIOFIX_NO_NUMBA"):
        raise ImportError("numba disabled by AUDIOFIX_NO_NUMBA")
    import numba
    _njit = numba.njit(cache=True)
    HAVE_NUMBA = True
except Exception:                                                # noqa: BLE001
    HAVE_NUMBA = False

    def _njit(f):
        return f


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(g):
    return 10 ** (np.asarray(g, dtype=np.float64) / 20)


def coef(ms, fs=SR):
    return float(np.exp(-1.0 / (fs * ms / 1000.0)))


# ------------------------------------------------------------------ envelopes
@_njit
def _follow(x, att, rel):
    out = np.empty_like(x)
    e = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        c = att if v > e else rel
        e = c * e + (1.0 - c) * v
        out[i] = e
    return out


def follow(x, att_ms, rel_ms, fs=SR):
    """Attack/release envelope follower of a non-negative signal."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if HAVE_NUMBA:
        return _follow(x, coef(att_ms, fs), coef(rel_ms, fs))
    w = max(1, int(fs * att_ms / 1000 * 3))
    r = coef(rel_ms, fs)
    return lfilter([1 - r], [1, -r], maximum_filter1d(x, size=w)).astype(np.float32)


@_njit
def _release(mn, rel):
    g = np.empty_like(mn)
    e = 1.0
    for i in range(mn.shape[0]):
        v = mn[i]
        if v < e:
            e = v
        else:
            e = rel * e + (1.0 - rel) * v
        g[i] = e
    return g


def lookahead_gain(req, la, rel_coef, hold=0):
    """Gain that never exceeds req: forward minimum over la samples, instant attack / exponential release,
    then a backward box average over la samples (a linear ramp that lands exactly on each peak)."""
    req = np.ascontiguousarray(req, dtype=np.float32)
    la = max(1, int(la))
    mn = minimum_filter1d(req, size=la + 1, origin=-((la + 1) // 2), mode="nearest")   # min over [i, i+la]
    if HAVE_NUMBA:
        g = _release(mn, np.float32(rel_coef))
    else:                                                                      # hold-then-smooth approximation
        h = max(la + 1, int(hold) or 1)
        g = minimum_filter1d(mn, size=h, origin=(h - 1) // 2, mode="nearest")
    return uniform_filter1d(g, size=la, origin=(la - 1) // 2, mode="nearest").astype(np.float32)  # mean over [i-la+1, i]


# ------------------------------------------------------------------ filters
def highpass(y, fc, order=4):
    return sosfiltfilt(butter(order, fc, btype="high", fs=SR, output="sos"), y, axis=0).astype(np.float32)


def lowpass(y, fc, order=4):
    return sosfiltfilt(butter(order, fc, btype="low", fs=SR, output="sos"), y, axis=0).astype(np.float32)


def bandpass(y, lo, hi, order=4):
    return sosfiltfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), y, axis=0).astype(np.float32)


def notch(y, f0, q=30.0):
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(f0, q, fs=SR)
    return filtfilt(b, a, y, axis=0).astype(np.float32)


def ramp_weights(n, edges, fade=1.5):
    """Raised-cosine membership weights for consecutive sections [(a, b), ...] in seconds; they sum to 1."""
    t = np.arange(n) / SR
    ws = []
    for i, (a, b) in enumerate(edges):
        w = np.ones(n, np.float32)
        if i > 0:
            u = np.clip((t - (a - fade / 2)) / fade, 0, 1)
            w *= (0.5 - 0.5 * np.cos(np.pi * u)).astype(np.float32)
        if i < len(edges) - 1:
            u = np.clip((t - (b - fade / 2)) / fade, 0, 1)
            w *= (0.5 + 0.5 * np.cos(np.pi * u)).astype(np.float32)
        ws.append(w)
    return ws


# ------------------------------------------------------------------ spectra and matching
CENT = 1000 * 2 ** (np.arange(-30, 27) / 6)                     # 1/6 octave, 31 Hz .. 20.2 kHz


def psd(y, nper=16384):
    f, pl = welch(y[:, 0], SR, nperseg=nper)
    _, pr = welch(y[:, 1], SR, nperseg=nper)
    return f, (pl + pr) / 2


def bands(f, P, cent=CENT):
    out = np.empty(len(cent))
    for i, c in enumerate(cent):
        s = (f >= c / 2 ** (1 / 12)) & (f < c * 2 ** (1 / 12))
        out[i] = P[s].mean() if s.any() else 1e-30
    return out


def ltas(y, ref_band=(500, 2000)):
    """Long-term average spectrum on the 1/6-octave grid, dB, normalised to the mean of ref_band."""
    f, P = psd(y)
    B = 10 * np.log10(bands(f, P) + 1e-30)
    sel = (CENT >= ref_band[0]) & (CENT <= ref_band[1])
    return B - B[sel].mean()


def band_rel(y, lo, hi, ref_to=None):
    """Power in [lo, hi) relative to the 500-2000 Hz power, dB."""
    f, P = psd(y)
    _, Q = psd(ref_to if ref_to is not None else y)
    return float(10 * np.log10(P[(f >= lo) & (f < hi)].mean() / (Q[(f >= 500) & (f <= 2000)].mean() + 1e-30) + 1e-30))


def match_curve(cur, ref, alpha, lo_db, hi_db, cutoff=None, low_limit=45.0, smooth=3):
    """Partial match of a 1/6-octave curve toward a reference, smoothed, clipped; no boosts where nothing exists."""
    d = uniform_filter1d(ref - cur, smooth, mode="nearest")
    w = np.full_like(d, alpha)
    lowz = CENT < low_limit
    w[lowz] *= np.clip((CENT[lowz] - low_limit * 2 / 3) / (low_limit / 3), 0, 1)
    g = np.clip(d * w, lo_db, hi_db)
    if cutoff:
        top = CENT > cutoff - 1000
        if (~top).any():
            g[top] = np.minimum(g[top], g[~top][-1])
    return g


def fir_from_curve(g_db, taps=8191, cent=CENT):
    f = np.concatenate([[0], cent, [SR / 2]])
    gg = np.concatenate([[g_db[0]], g_db, [g_db[-1]]])
    grid = np.linspace(0, SR / 2, 4097)
    resp = undb(np.interp(np.log(grid + 1), np.log(f + 1), gg))
    return firwin2(taps, grid, resp, fs=SR).astype(np.float32)


def apply_fir(y, h):
    pad = len(h) // 2
    out = np.stack([oaconvolve(y[:, c], h, mode="full")[pad:pad + len(y)] for c in range(y.shape[1])], axis=1)
    return out.astype(np.float32)


def apply_sectioned(y, filters, weights):
    """Sum over sections of weight_i * FIR_i(y), filtering only where each weight is non-zero."""
    out = np.zeros_like(y)
    for h, w in zip(filters, weights):
        nz = np.flatnonzero(w > 1e-4)
        if not len(nz):
            continue
        pad = len(h)
        a, b = max(0, nz[0] - pad), min(len(y), nz[-1] + 1 + pad)
        out[a:b] += apply_fir(y[a:b], h) * w[a:b, None]
    return out


# ------------------------------------------------------------------ dynamics
def isp_peaks(y, block=SR * 30):
    """Per-sample inter-sample-aware peak: max |x| over the 4x oversampled points around each sample."""
    n = len(y)
    out = np.empty(n, np.float32)
    P = 64
    for a in range(0, n, block):
        b = min(n, a + block)
        a0, b0 = max(0, a - P), min(n, b + P)
        up = np.abs(resample_poly(y[a0:b0], 4, 1, axis=0)).max(axis=1)
        m = (b0 - a0)
        pk = up[:m * 4].reshape(m, 4).max(axis=1)
        out[a:b] = np.maximum(pk[a - a0:a - a0 + (b - a)], np.abs(y[a:b]).max(axis=1))
    return out


def limiter_gain(peaks, ceiling_db, look_ms=1.5, release_ms=120.0, pre_gain_db=0.0):
    c = float(undb(ceiling_db - 0.1))
    req = np.minimum(1.0, c / np.maximum(peaks * undb(pre_gain_db), 1e-9)).astype(np.float32)
    la = int(SR * look_ms / 1000)
    return lookahead_gain(req, la, coef(release_ms), hold=int(SR * release_ms / 2000))


def limit(y, ceiling_db=-1.0, look_ms=1.5, release_ms=120.0, peaks=None):
    """Stereo-linked lookahead limiter with inter-sample-aware detection. Returns (out, stats)."""
    pk = isp_peaks(y) if peaks is None else peaks
    g = limiter_gain(pk, ceiling_db, look_ms, release_ms)
    gr = -db(np.maximum(g, 1e-6))
    stats = {"max_gr_db": float(gr.max()), "over1_pct": float(np.mean(gr > 1) * 100),
             "over2_pct": float(np.mean(gr > 2) * 100), "active_pct": float(np.mean(gr > 0.01) * 100)}
    return (y * g[:, None]).astype(np.float32), stats


def compress(y, ratio, att_ms, rel_ms, pct=60, knee_db=6.0, max_gr=12.0, active_floor_db=-60.0):
    """Feed-forward, stereo-linked compressor. The threshold sits at a percentile of the detector level on
    active passages; make-up gain restores the RMS, so only the shape of the level changes."""
    det = follow(np.abs(y).max(axis=1), att_ms, rel_ms)
    lvl = db(det + 1e-9)
    act = lvl > active_floor_db
    if not act.any():
        return y, {"threshold_db": None, "mean_gr_db": 0.0, "max_gr_db": 0.0}
    thr = float(np.percentile(lvl[act], pct))
    over = lvl - thr
    k = 1 - 1 / ratio
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * k, k * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    gr = follow(np.minimum(gr, max_gr).astype(np.float32), 1.0, rel_ms / 2)
    out = y * undb(-gr)[:, None]
    mk = np.sqrt(np.mean(y.astype(np.float64) ** 2) / (np.mean(out.astype(np.float64) ** 2) + 1e-20))
    return (out * mk).astype(np.float32), {"threshold_db": thr, "mean_gr_db": float(gr[act].mean()),
                                           "max_gr_db": float(gr.max())}


def deess(v, lo=4500, hi=11000, pct=80, ratio=3.0, max_cut=9.0):
    """Dynamic de-esser: only the 4.5-11 kHz band of the louder-than-usual sibilant moments is reduced."""
    band = bandpass(v, lo, hi)
    det_b = follow(np.abs(band).max(axis=1), 1.0, 60)
    det_f = follow(np.abs(v).max(axis=1), 5.0, 120)
    rel = db(det_b + 1e-9) - db(det_f + 1e-9)
    act = det_f > undb(-50)
    if not act.any():
        return v, {}
    thr = float(np.percentile(rel[act], pct))
    gr = np.clip((rel - thr) * (1 - 1 / ratio), 0, max_cut).astype(np.float32)
    gr = follow(gr, 0.5, 40)
    out = v - band + band * undb(-gr)[:, None]
    on = gr > 0.1
    return out.astype(np.float32), {"threshold_db": thr, "mean_cut_db": float(gr[on].mean()) if on.any() else 0.0,
                                    "max_cut_db": float(gr.max())}


def transients(d, amount=0.35, max_db=2.5):
    """Lift drum hits: the dB difference between a fast and a slow envelope, scaled and capped."""
    det = np.abs(d).max(axis=1)
    r = db(follow(det, 0.3, 40) + 1e-7) - db(follow(det, 15, 150) + 1e-7)
    g = follow(np.clip(r * amount, 0, max_db).astype(np.float32), 0.3, 30)
    on = g > 0.5
    return (d * undb(g)[:, None]).astype(np.float32), {"mean_db": float(g[on].mean()) if on.any() else 0.0,
                                                        "max_db": float(g.max())}


def widen(y, side_db=3.0, fc=150):
    m, s = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
    s = s + sosfiltfilt(butter(2, fc, btype="high", fs=SR, output="sos"), s) * (undb(side_db) - 1)
    return np.stack([m + s, m - s], 1).astype(np.float32)


def mono_bass(y, fc=100):
    m, s = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
    s = sosfiltfilt(butter(4, fc, btype="high", fs=SR, output="sos"), s)
    return np.stack([m + s, m - s], 1).astype(np.float32)


# ------------------------------------------------------------------ stereo repair
def _antiphase_block(y, nper=4096, hop=1024):
    _, _, L = stft(y[:, 0], SR, window="hann", nperseg=nper, noverlap=nper - hop)
    _, _, R = stft(y[:, 1], SR, window="hann", nperseg=nper, noverlap=nper - hop)
    C = L * np.conj(R)
    sm = lambda a: uniform_filter(a, size=(5, 43), mode="nearest")
    cr, ci = sm(C.real.astype(np.float32)), sm(C.imag.astype(np.float32))
    pl, pr = sm((np.abs(L) ** 2).astype(np.float32)), sm((np.abs(R) ** 2).astype(np.float32))
    mag = np.sqrt(cr ** 2 + ci ** 2) + 1e-20
    coh = mag ** 2 / (pl * pr + 1e-30)
    cos = cr / mag
    anti = np.clip((-cos - 0.15) / 0.55, 0, 1) * np.clip((coh - 0.12) / 0.33, 0, 1)
    anti = uniform_filter1d(anti, 21, axis=1)
    _, r2 = istft(R * np.exp(1j * np.pi * anti).astype(np.complex64), SR, window="hann", nperseg=nper,
                  noverlap=nper - hop)
    w = np.abs(L) ** 2 + np.abs(R) ** 2
    return r2[:len(y)].astype(np.float32), float((anti * w).sum()), float(w.sum())


def fix_antiphase(y, block_s=60.0, overlap_s=4.0):
    """Turn R toward L wherever both channels carry the same component in anti-phase (coherent, cos phase < 0).
    Unit-modulus rotation per STFT bin, statistics smoothed over ~1 s and +-2 bins; ordinary stereo is left alone.
    Processed in overlapping blocks so memory stays bounded on long files."""
    n = len(y)
    B, O = int(block_s * SR), int(overlap_s * SR)
    acc = np.zeros(n, np.float64)
    wsum = np.zeros(n, np.float64)
    treated = total = 0.0
    a = 0
    while True:
        b = min(n, a + B)
        r2, t, tot = _antiphase_block(y[a:b])
        w = np.ones(b - a)
        if a > 0:
            w[:O] = np.linspace(0, 1, O)
        if b < n:
            w[-O:] = np.linspace(1, 0, O)
        acc[a:b] += r2 * w
        wsum[a:b] += w
        treated += t; total += tot
        if b == n:
            break
        a = b - O
    out = y.copy()
    out[:, 1] = (acc / np.maximum(wsum, 1e-9)).astype(np.float32)
    return out, {"treated_pct": 100 * treated / max(total, 1e-30)}


# ------------------------------------------------------------------ band extension
def hf_extend(y, cutoff, target_rel_db, top=19500.0, rolloff=17500.0, mask=None):
    """SBR-style extension above a codec cut-off: the band just below the cut-off is shifted up
    (single-sideband, so hats, cymbals and sibilants keep their timing), repeated until ~top, then set so that
    the band right above the cut-off sits at target_rel_db (power re 0.5-2 kHz). mask: samples to calibrate on."""
    from scipy.fft import next_fast_len
    from scipy.signal import hilbert
    n = len(y)
    nf = next_fast_len(n)
    t = np.arange(n) / SR
    ext = np.zeros_like(y)
    fc = float(cutoff)
    passes = 0
    while fc < top - 400 and passes < 4:
        delta = min(fc * 0.25, top - fc)
        src = bandpass(y + ext, fc - delta, fc - 150, order=8)
        car = np.exp(2j * np.pi * delta * t)
        sh = np.empty_like(y)
        for c in range(2):
            sh[:, c] = np.real(hilbert(src[:, c], N=nf)[:n] * car)
        sh = highpass(sh, fc - 100, order=8)
        ext += sh
        fc += delta
        passes += 1
    if passes == 0:
        return y, {"gain_db": None}
    if rolloff and rolloff > cutoff:
        ext = lowpass(ext, rolloff, order=2)
    lo, hi = cutoff + 300, min(cutoff + 2500, 18000)
    have = band_rel(ext, lo, hi, ref_to=y) if mask is None else band_rel(ext[mask], lo, hi, ref_to=y[mask])
    g = float(10 ** ((target_rel_db - have) / 20))
    return (y + ext * g).astype(np.float32), {"gain_db": 20 * np.log10(g), "passes": passes,
                                               "band": [lo, hi], "target_rel_db": target_rel_db}
