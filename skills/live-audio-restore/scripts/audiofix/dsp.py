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


def notches(y, freqs, bw_hz=15.0):
    """Notch every frequency in freqs, bw_hz wide each, as one zero-phase cascade."""
    from scipy.signal import iirnotch, sosfiltfilt, tf2sos
    freqs = [f for f in freqs if 0 < f < SR / 2 - bw_hz]
    if not freqs:
        return y
    sos = np.concatenate([tf2sos(*iirnotch(f, f / bw_hz, fs=SR)) for f in freqs])
    return sosfiltfilt(sos, y, axis=0).astype(np.float32)


def sbr_cut(m, shifts, nperseg=4096, hop=1024, min_db=6.0, max_cut_db=30.0):
    """How many dB to take off each STFT bin of the top band so that the tones spectral band replication copied up
    (measure.sbr_copies) sink to the level of their neighbourhood.

    A bin is cut in a frame where it peaks min_db or more over its +-150 Hz median and the low band, shift_hz below,
    peaks there too (the tone it was copied from). Every bin of the peak (+-2 bins, the window's main lobe) goes down
    to the median, by at most max_cut_db, held one frame either side. Noise and envelope stay. Returns (k0, cut), cut
    in dB with cut[k - k0, frame], computed on the mono mix and applied alike to the mix and every stem."""
    f, _, Z = stft(m, SR, nperseg=nperseg, noverlap=nperseg - hop)
    df = f[1] - f[0]
    bands = [(int(round(s["shift_hz"] / df)), int(s["band_hz"][0] / df), int(s["band_hz"][1] / df) + 1)
             for s in shifts]
    k_lo, k_hi = min(b[1] for b in bands) - 2, max(b[2] for b in bands) + 2
    r0 = max(0, min(b[1] - b[0] for b in bands) - 20)
    r1 = min(len(f), k_hi + 20)
    A = 20 * np.log10(np.abs(Z[r0:r1]) + 1e-9).astype(np.float32)
    del Z
    from scipy.ndimage import median_filter
    E = A - median_filter(A, size=(29, 1))
    cut = np.zeros((k_hi - k_lo, E.shape[1]), np.float32)
    for D, a, b in bands:
        for k in range(max(a, D + 2), min(b, r1 - 2)):
            e = E[k - r0]
            hit = (e >= min_db) & (e >= E[k - 1 - r0]) & (e >= E[k + 1 - r0])
            if not hit.any():
                continue
            src = np.maximum(np.maximum(E[k - D - 1 - r0], E[k - D - r0]), E[k - D + 1 - r0])
            hit &= src >= min_db
            if not hit.any():
                continue
            for j in range(-2, 3):
                c = np.where(hit, np.clip(E[k + j - r0], 0, max_cut_db), 0)
                cut[k + j - k_lo] = np.maximum(cut[k + j - k_lo], c)
    return k_lo, maximum_filter1d(cut, 3, axis=1)


def apply_cut(y, k0, cut, nperseg=4096, hop=1024):
    """Apply an STFT-domain cut (dB, rows from bin k0) to every channel; the rest of the spectrum passes untouched."""
    g = (10 ** (-cut / 20)).astype(np.float32)
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        _, _, Z = stft(y[:, c], SR, nperseg=nperseg, noverlap=nperseg - hop)
        Z[k0:k0 + len(g)] *= g[:, :Z.shape[1]]
        _, x = istft(Z, SR, nperseg=nperseg, noverlap=nperseg - hop)
        out[:, c] = x[:len(y)]
    return out


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
def noise_like(sig, nper=1024, smooth_bins=9, seed=1234):
    """Same time-frequency envelope as sig, noise-like fine structure: STFT magnitude smoothed over frequency, random
    phase shared by both channels (so the panning stays). A shifted codec band is sparse and peaky ("metallic");
    real air from cymbals and breath is dense and noise-like."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(sig)
    ph = None
    for c in range(sig.shape[1]):
        f, t, Z = stft(sig[:, c], SR, nperseg=nper, noverlap=nper * 3 // 4)
        if ph is None:
            ph = np.exp(2j * np.pi * rng.random(Z.shape))
        mag = uniform_filter1d(np.abs(Z), smooth_bins, axis=0)
        _, z = istft(mag * ph, SR, nperseg=nper, noverlap=nper * 3 // 4)
        z = z[:len(sig)]
        out[:, c] = np.pad(z, (0, len(sig) - len(z))) if len(z) < len(sig) else z
    return out


def _ssb_copy(y, start, top=19500.0):
    """The band just below `start` shifted up (single-sideband, so hats, cymbals and sibilants keep their timing),
    repeated until about `top`. Returns (copy, passes)."""
    from scipy.fft import next_fast_len
    from scipy.signal import hilbert
    n = len(y)
    nf = next_fast_len(n)
    t = np.arange(n) / SR
    ext = np.zeros_like(y)
    fc = float(start)
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
    return ext, passes


def _psd2(y, nper=4096):
    f, pl = welch(y[:, 0], SR, nperseg=min(nper, len(y)))
    _, pr = welch(y[:, 1], SR, nperseg=min(nper, len(y)))
    return f, (pl + pr) / 2


def hf_fill(y, start, rel_db_at, top=19500.0, below_db=0.0, noise_mix=0.5, section_s=20.0, band_hz=250.0,
            max_gain_db=12.0, taps=4095):
    """Rebuild the top above `start` toward the reference's spectrum, adding only what is missing.

    The material is the SSB copy (half of it noise-like, as real air is dense). Then, per section (crossfaded, so a
    codec whose cut-off moves over time is followed) and per band_hz band, the copy is scaled so that the existing
    content plus the copy reaches the target: rel_db_at(f) is the reference's level at f re its 0.5-2 kHz level, applied
    to the section's own 0.5-2 kHz level, minus below_db. Bands that already hold enough get nothing.

    The single-gain extension it replaces calibrated one band above the cut-off: the copies above it followed the
    source band, not the reference's roll-off (16-19 kHz came out 5-10 dB over the studio version, heard as a digital,
    scattered vocal), and a start above a moving cut-off left a 10-13 dB notch."""
    n = len(y)
    ext, passes = _ssb_copy(y, start, top)
    if passes == 0:
        return y, {"applied": False}
    if noise_mix > 0:
        nz = highpass(noise_like(ext), start - 100, order=8)
        p_e, p_n = float(np.mean(ext.astype(np.float64) ** 2)), float(np.mean(nz.astype(np.float64) ** 2)) + 1e-30
        ext = (np.sqrt(1 - noise_mix) * ext + np.sqrt(noise_mix * p_e / p_n) * nz).astype(np.float32)
    dur = n / SR
    k = max(1, int(round(dur / section_s)))
    edges = np.linspace(0.0, dur, k + 1)
    secs = [[float(edges[i]), float(edges[i + 1])] for i in range(k)]
    W = ramp_weights(n, secs, fade=2.0)
    lows = np.arange(float(start), top, band_hz)
    centres = lows + band_hz / 2
    tgt = np.array([rel_db_at(c) for c in centres]) - below_db
    cap = 10 ** (max_gain_db / 20)
    firs, used = [], []
    for a, b in secs:
        i0, i1 = int(a * SR), int(b * SR)
        f, py = _psd2(y[i0:i1])
        _, pe = _psd2(ext[i0:i1])
        ref = py[(f >= 500) & (f <= 2000)].mean()
        g = np.zeros(len(centres))
        if ref > 1e-14:
            need = np.array([ref * 10 ** (tgt[j] / 10) - py[(f >= lo) & (f < lo + band_hz)].mean()
                             for j, lo in enumerate(lows)])
            e = np.array([pe[(f >= lo) & (f < lo + band_hz)].mean() for lo in lows])
            # a band where the copy is weak (a pass edge) is scaled as if it held a quarter of its neighbours' energy:
            # a huge gain there only magnified a stray tone into a 19 kHz whistle
            e = np.maximum(e, uniform_filter1d(e, 5, mode="nearest") / 4)
            ok = (need > 0) & (e > 1e-30)
            g[ok] = np.minimum(np.sqrt(need[ok] / e[ok]), cap)
        g = np.convolve(g, np.ones(3) / 3, mode="same")                  # no band switches on or off alone
        used.append(g)
        freq = np.concatenate([[0.0, start - band_hz], centres, [top + band_hz, SR / 2]])
        gain = np.concatenate([[0.0, 0.0], g, [0.0, 0.0]])
        firs.append(firwin2(taps, freq, gain, fs=SR).astype(np.float32))
    out = (y + apply_sectioned(ext, firs, W)).astype(np.float32)
    f, po = _psd2(out, 8192)
    ref = po[(f >= 500) & (f <= 2000)].mean()
    dev = {}
    for c0 in range(int(np.ceil(start / 1000.0) * 1000), int(top), 1000):
        s = (f >= c0) & (f < min(c0 + 1000, top))
        dev[str(c0)] = round(float(10 * np.log10(po[s].mean() / ref + 1e-30)
                                   - (rel_db_at((c0 + min(c0 + 1000, top)) / 2) - below_db)), 1)
    return out, {"applied": True, "start_hz": round(float(start)), "passes": passes, "sections": k,
                 "bands_filled_pct": round(float(np.mean(np.array(used) > 1e-3) * 100), 1),
                 "result_minus_target_db": dev}


def hf_extend(y, cutoff, target_rel_db, top=19500.0, rolloff=17500.0, mask=None, noise_mix=0.0):
    """SBR-style extension above a codec cut-off: the band just below the cut-off is shifted up
    (single-sideband, so hats, cymbals and sibilants keep their timing), repeated until ~top, then set so that
    the band right above the cut-off sits at target_rel_db (power re 0.5-2 kHz). mask: samples to calibrate on.
    noise_mix: share of the extension's power replaced by a noise-like copy with the same envelope (0..1)."""
    ext, passes = _ssb_copy(y, cutoff, top)
    if passes == 0:
        return y, {"gain_db": None}
    if rolloff and rolloff > cutoff:
        ext = lowpass(ext, rolloff, order=2)
    if noise_mix > 0:
        nz = highpass(noise_like(ext), cutoff - 100, order=8)
        p_e, p_n = float(np.mean(ext.astype(np.float64) ** 2)), float(np.mean(nz.astype(np.float64) ** 2)) + 1e-30
        ext = (np.sqrt(1 - noise_mix) * ext + np.sqrt(noise_mix * p_e / p_n) * nz).astype(np.float32)
    lo, hi = cutoff + 300, min(cutoff + 2500, 18000)
    have = band_rel(ext, lo, hi, ref_to=y) if mask is None else band_rel(ext[mask], lo, hi, ref_to=y[mask])
    g = float(10 ** ((target_rel_db - have) / 20))
    return (y + ext * g).astype(np.float32), {"gain_db": 20 * np.log10(g), "passes": passes,
                                               "band": [lo, hi], "target_rel_db": target_rel_db}
