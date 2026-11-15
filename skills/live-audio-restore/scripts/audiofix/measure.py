"""Diagnostic measurements. Each function returns plain numbers so the plan rules and the report can use them."""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, csd, sosfiltfilt, welch

from . import SR
from . import loudness as L
from .dsp import CENT, bandpass, db, ltas, psd

BANDS6 = [(40, 120), (120, 300), (300, 1000), (1000, 3000), (3000, 8000), (8000, 15500)]


def rms_db(y, win_s=0.4, hop_s=0.1):
    """Short-time RMS level (dBFS) of the channel mean, for gap / level analysis."""
    m = y.mean(axis=1).astype(np.float64)
    w, h = int(win_s * SR), int(hop_s * SR)
    c = np.concatenate([[0.0], np.cumsum(m ** 2)])
    st = np.arange(0, max(1, len(m) - w + 1), h)
    return db(np.sqrt((c[st + w] - c[st]) / w) + 1e-12), st / SR + win_s / 2


def clipping(y):
    """Flat-topped runs near the peak (digital clipping) per minute, and how crowded the top 0.3 dB is."""
    a = np.abs(y).max(axis=1)
    pk = float(a.max())
    near = a >= pk * 0.966                                      # within 0.3 dB of the peak
    runs = 0
    if pk > 0.5:
        d = np.abs(np.diff(y, axis=0)).max(axis=1) < 1e-4
        flat = near[1:] & d
        edges = np.flatnonzero(np.diff(np.concatenate([[0], flat.astype(np.int8), [0]])))
        lengths = edges[1::2] - edges[::2]
        runs = int((lengths >= 2).sum())
    minutes = len(y) / SR / 60
    return {"peak_dbfs": round(float(db(pk)), 2), "flat_runs_per_min": round(runs / max(minutes, 1e-9), 1),
            "near_peak_pct": round(float(near.mean() * 100), 4)}


def dc_offset(y):
    return [round(float(db(abs(y[:, c].mean()) + 1e-12)), 1) for c in range(2)]


def hum(y):
    """Narrow mains peaks at 50 or 60 Hz and harmonics, dB above the local spectrum."""
    f, P = welch(y.mean(axis=1), SR, nperseg=1 << 16)
    Pd = 10 * np.log10(P + 1e-30)
    out = {}
    for base in (50, 60):
        ex = []
        for k in range(1, 7):
            fk = base * k
            pk = Pd[(f > fk - 1.5) & (f < fk + 1.5)].max()
            nb = Pd[((f > fk - 12) & (f < fk - 4)) | ((f > fk + 4) & (f < fk + 12))]
            ex.append(float(pk - np.median(nb)))
        out[base] = ex
    best = max(out, key=lambda b: sum(sorted(out[b])[-3:]))
    ex = out[best]
    present = ex[0] >= 10 or sum(e >= 8 for e in ex) >= 2
    return {"mains_hz": best, "excess_db": [round(e, 1) for e in ex], "present": bool(present)}


def bandwidth(y):
    """Highest frequency with real content, and whether it ends in a codec-style brick wall."""
    f, P = welch(y.mean(axis=1), SR, nperseg=8192)
    Pd = uniform_filter1d(10 * np.log10(P + 1e-30), 9)
    ref = np.median(Pd[(f >= 1000) & (f <= 4000)])
    alive = np.flatnonzero(Pd > ref - 45)
    fc = float(f[alive[-1]]) if len(alive) else float(SR / 2)
    below = Pd[(f > fc - 900) & (f < fc - 300)].mean() if fc > 1000 else ref
    above = Pd[(f > fc + 300) & (f < fc + 900)].mean() if fc < SR / 2 - 1000 else below
    return {"cutoff_hz": round(fc), "edge_drop_db": round(float(below - above), 1),
            "brick_wall": bool(below - above > 25 and fc < 20000)}


def codec_notches(y, lo=3000, hi=15500):
    """Narrow, persistent dips well below the smoothed spectrum (bands a lossy codec zeroed)."""
    f, P = welch(y.mean(axis=1), SR, nperseg=8192)
    Pd = 10 * np.log10(P + 1e-30)
    smooth = median_filter(Pd, size=int(4000 / (SR / 8192)) | 1)
    dip = smooth - uniform_filter1d(Pd, 5)
    sel = (f >= lo) & (f <= hi) & (dip > 4)
    out = []
    idx = np.flatnonzero(sel)
    if len(idx):
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
        merged = [groups[0]]
        for g in groups[1:]:
            if f[g[0]] - f[merged[-1][-1]] < 150:
                merged[-1] = np.concatenate([merged[-1], g])
            else:
                merged.append(g)
        for g in merged:
            if f[g[-1]] - f[g[0]] >= 150:
                out.append({"from_hz": round(float(f[g[0]])), "to_hz": round(float(f[g[-1]])),
                            "depth_db": round(float(dip[g].max()), 1)})
    return out[:8]


def stereo(y):
    """Mono / near-mono / stereo, per-band L-R correlation, side-to-mid level, mono loss, anti-phase timeline."""
    side = (y[:, 0] - y[:, 1]) / 2
    mid = (y[:, 0] + y[:, 1]) / 2
    sm = float(db(np.std(side) / (np.std(mid) + 1e-12)))
    kind = "mono" if sm < -45 else "near-mono" if sm < -25 else "stereo"
    res = {"kind": kind, "side_minus_mid_db": round(sm, 1)}
    if kind == "mono":
        return res
    bands = {}
    for lo, hi in BANDS6:
        if hi >= SR / 2:
            continue
        b = bandpass(y, lo, hi)
        bands[f"{lo}-{hi}"] = round(float(np.corrcoef(b[:, 0], b[:, 1])[0, 1]), 2)
    res["band_corr"] = bands
    mid_band = bandpass(y, 300, 3000)
    w = 4 * SR
    tl = []
    for i in range(0, len(y) - w + 1, w // 2):
        s = mid_band[i:i + w]
        if np.std(s) > 1e-5:
            tl.append(float(np.corrcoef(s[:, 0], s[:, 1])[0, 1]))
    tl = np.array(tl) if tl else np.zeros(1)
    res["mid_corr_min"] = round(float(tl.min()), 2)
    res["mid_corr_mean"] = round(float(tl.mean()), 2)
    res["mid_corr_neg_pct"] = round(float(np.mean(tl < -0.2) * 100), 1)
    st = L.integrated(y)
    mono = L.integrated(np.repeat(mid[:, None], 2, axis=1))
    res["mono_loss_db"] = round(mono - st, 2)
    return res


def coherent_antiphase(stem, max_s=120.0, chunk_s=30.0):
    """Share of a stem's energy that the polarity repair would rotate (coherent anti-phase STFT bins), estimated on
    up to max_s seconds spread over the stem, plus the stem's mono-sum loss. Catches anti-phase confined to some bands
    (a reversed low end, a phasey vocal), which the 4 s window correlation of stem_stereo misses."""
    from .dsp import fix_antiphase
    n, L = len(stem), int(chunk_s * SR)
    k = max(1, int(max_s // chunk_s))
    if n <= k * L:
        sample = stem
    else:
        sample = np.concatenate([stem[s:s + L] for s in np.linspace(0, n - L, k).astype(int)])
    _, a = fix_antiphase(sample, block_s=2 * chunk_s, overlap_s=2.0)
    return {"coherent_anti_pct": round(float(a["treated_pct"]), 1),
            "mono_loss_db": stereo(stem).get("mono_loss_db", 0.0)}


def stem_stereo(stem, mix_energy=None, win_s=4.0):
    """Anti-phase check for one stem: energy-weighted share of windows with L-R correlation < -0.3,
    plus the phase picture per band over those windows."""
    w = int(win_s * SR)
    cs, es, starts = [], [], []
    for i in range(0, len(stem) - w + 1, w // 2):
        s = stem[i:i + w]
        e = float(np.mean(s.astype(np.float64) ** 2))
        if e < 1e-10:
            continue
        cs.append(float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])); es.append(e); starts.append(i)
    if not cs:
        return {"anti_pct": 0.0}
    cs, es = np.array(cs), np.array(es)
    anti = cs < -0.3
    res = {"anti_pct": round(float(es[anti].sum() / es.sum() * 100), 1), "corr_min": round(float(cs.min()), 2),
           "corr_median": round(float(np.median(cs)), 2)}
    if anti.any():
        segs = np.concatenate([stem[starts[k]:starts[k] + w] for k in np.flatnonzero(anti)])
        f, Pxy = csd(segs[:, 0], segs[:, 1], fs=SR, nperseg=4096)
        _, Pxx = welch(segs[:, 0], SR, nperseg=4096)
        _, Pyy = welch(segs[:, 1], SR, nperseg=4096)
        cos = np.cos(np.angle(Pxy))
        bands = {}
        for lo, hi in [(60, 150), (150, 400), (400, 1000), (1000, 2500), (2500, 6000), (6000, 15000)]:
            sel = (f >= lo) & (f < hi)
            wt = Pxx[sel] + Pyy[sel]
            bands[f"{lo}-{hi}"] = round(float(np.average(cos[sel], weights=wt)), 2)
        res["anti_band_cos"] = bands
    return res


def treble_share(stems, lo=3000, hi=10000):
    """Which stem carries the 3-10 kHz energy (where harshness lives)."""
    out = {}
    for name, s in stems.items():
        f, P = psd(s)
        out[name] = float(P[(f >= lo) & (f < hi)].sum())
    tot = sum(out.values()) + 1e-30
    return {k: round(v / tot * 100) for k, v in out.items()}


def level_re_mix(stem, mix, mask=None):
    s = stem if mask is None else stem[mask]
    m = mix if mask is None else mix[mask]
    return round(float(10 * np.log10(np.mean(s.astype(np.float64) ** 2) / (np.mean(m.astype(np.float64) ** 2) + 1e-30)
                                     + 1e-30)), 1)


def _close_and_clean(act, hop_s, fill_s=0.35, min_s=0.1):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(np.int8), [0]])))
    for s0, e0 in zip(idx[1:-1:2], idx[2::2]):                    # pauses between bursts
        if e0 - s0 < fill_s / hop_s:
            act[s0:e0] = True
    idx = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(np.int8), [0]])))
    for s0, e0 in zip(idx[::2], idx[1::2]):
        if e0 - s0 < min_s / hop_s:
            act[s0:e0] = False
    return act


def harmonic_activity(voice, hop_s=0.01, win_frames=25, threshold=0.35):
    """Talker detector without ASR: voiced speech is periodic (pitch 80-400 Hz), crowd shouting much less.
    Smoothed autocorrelation peak over 250 ms, 125 ms hangover. About F1 0.8 against word timings on live MC."""
    from scipy.signal import fftconvolve
    vb = bandpass(voice, 250, 3800).mean(axis=1).astype(np.float32)
    h = int(hop_s * SR); W = int(0.04 * SR)
    n = max(0, (len(vb) - W) // h)
    lev = np.full(n, -120.0); harm = np.zeros(n)
    lo, hi = int(SR / 400), int(SR / 80)
    for i in range(n):
        seg = vb[i * h:i * h + W]
        e = float(np.sqrt(np.mean(seg ** 2)))
        lev[i] = 20 * np.log10(e + 1e-12)
        if e < 1e-5:
            continue
        ac = fftconvolve(seg, seg[::-1], mode="full")[W - 1:]
        harm[i] = ac[lo:hi].max() / (ac[0] + 1e-12)
    loud = lev > max(np.percentile(lev, 30), lev.max() - 50)
    act = (uniform_filter1d(harm, win_frames) > threshold) & loud
    act = uniform_filter1d(act.astype(np.float32), 25) > 0
    return _close_and_clean(act, hop_s), {"method": "harmonic", "threshold": threshold}


def asr_available():
    import os
    if os.environ.get("AUDIOFIX_NO_ASR"):
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def asr_activity(voice, regions=None, model_name=None, language=None, hop_s=0.01, beam_size=1, strict=True,
                 seed=None, max_compression=None):
    """Talker detector from Whisper word timings (faster-whisper): only intelligible words count, so crowd
    shouting does not. With strict=True (locating the talker), known applause hallucinations and low-confidence
    words are dropped. verify uses strict=False, beam search and seed=0: every word is kept and weighed by its
    probability inside a mask that is already known. Whisper's temperature fallback, which re-decodes windows where
    decoding loops (a cheer came back as "oh-oh-oh..." 110 times without it), samples at random and moves word
    counts by several % between identical runs; a fixed seed makes it reproducible. max_compression=2.4 (Whisper's
    own loop test) drops a loop that survives every fallback temperature. Also returns each word's log-probability."""
    import os
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    name = model_name or os.environ.get("AUDIOFIX_ASR_MODEL", "small")
    wm = WhisperModel(name, device="cpu", compute_type="int8")
    n = len(voice)
    act = np.zeros(int(n / SR / hop_s) + 1, bool)
    regions = regions or [(0.0, n / SR)]
    probs, times = [], []
    lang = language or os.environ.get("AUDIOFIX_ASR_LANG") or None
    chunks = []
    for a, b in regions:
        seg = voice[int(a * SR):int(b * SR)].mean(axis=1)
        if len(seg) >= SR:
            chunks.append((a, resample_poly(seg, 160, 441).astype(np.float32)))
    lid = None
    if lang is None and chunks:
        # Language ID alone is fooled by degraded voices; decode a sample with the 3 likeliest languages and keep the
        # one that Whisper itself finds most probable (mean segment log-prob weighted by duration).
        allx = np.concatenate([c for _, c in chunks])[:16000 * 120]
        try:
            _, _, allp = wm.detect_language(allx, language_detection_segments=4)
            cands = [l for l, _ in allp[:3]]
        except Exception:                              # noqa: BLE001 (older faster-whisper)
            _, inf = wm.transcribe(allx[:16000 * 30], beam_size=1)
            cands = [inf.language]
        sample, scores = allx[:16000 * 60], {}
        for L in cands:
            sg = list(wm.transcribe(sample, language=L, beam_size=1, condition_on_previous_text=False)[0])
            dur = sum(max(x.end - x.start, 0.01) for x in sg)
            scores[L] = sum(x.avg_logprob * max(x.end - x.start, 0.01) for x in sg) / dur if sg else -9.0
        lang = max(scores, key=scores.get)
        lid = {k: round(v, 3) for k, v in scores.items()}
    loops = 0
    for a, x16 in chunks:
        if seed is not None:
            import ctranslate2
            ctranslate2.set_random_seed(seed)                # per chunk, so a chunk's result does not depend on others
        segs, info = wm.transcribe(x16, language=lang, word_timestamps=True, beam_size=beam_size, vad_filter=False,
                                   condition_on_previous_text=False)
        for sg in list(segs):                                # decode now, right after seeding
            if strict and (sg.no_speech_prob > 0.6 or sg.avg_logprob < -1.0):
                continue
            if max_compression is not None and sg.compression_ratio > max_compression:
                loops += 1
                continue
            for w in sg.words or []:
                if strict and w.probability < 0.15:
                    continue
                probs.append(np.log(max(w.probability, 1e-6)))
                times.append(round(a + (w.start + w.end) / 2, 2))
                act[int(max(0, a + w.start - 0.12) / hop_s):int((a + w.end + 0.25) / hop_s)] = True
    return _close_and_clean(act, hop_s), {"method": "asr", "model": name, "language": lang, "language_scores": lid,
                                          "words": len(probs), "mean_word_logprob": float(np.mean(probs)) if probs else None,
                                          "word_times": times, "word_logprobs": [round(float(p), 4) for p in probs],
                                          "loop_segments_dropped": loops}


def speech_activity(voice, regions=None, method="auto", hop_s=0.01):
    """Frame mask (10 ms) of the MC talker. method: asr | harmonic | auto (asr when faster-whisper is installed)."""
    if method in ("auto", "asr") and asr_available():
        try:
            return asr_activity(voice, regions, hop_s=hop_s)
        except Exception as e:                                   # noqa: BLE001
            act, info = harmonic_activity(voice, hop_s)
            info["asr_error"] = str(e)[:200]
            return act, info
    return harmonic_activity(voice, hop_s)


def speech_metrics(mix, vocals, act, hop_s):
    """For spoken passages: crowd/noise level in the gaps relative to speech, speech-to-background ratio,
    low-frequency rumble, and the bandwidth of the voice itself."""
    h = int(hop_s * SR)
    n = min(len(act), len(mix) // h)
    act = act[:n]
    lv, t = rms_db(mix, 0.4, hop_s)
    lv = lv[:n] if len(lv) >= n else np.pad(lv, (0, n - len(lv)), mode="edge")
    res = {"speech_pct": round(float(act.mean() * 100), 1)}
    if act.any() and (~act).any():
        res["gap_minus_speech_db"] = round(float(np.median(lv[~act]) - np.median(lv[act])), 1)
        res["gap_p90_minus_speech_db"] = round(float(np.percentile(lv[~act], 90) - np.median(lv[act])), 1)
    mask = np.repeat(act, h)
    mask = np.concatenate([mask, np.zeros(len(mix) - len(mask), bool)])
    if mask.any():
        if vocals is not None:
            bg = mix[mask] - vocals[mask]
            res["speech_to_background_db"] = round(float(db(np.std(vocals[mask]) / (np.std(bg) + 1e-12))), 1)
        else:
            vocals = mix
        f, P = psd(vocals[mask])
        lowr = P[(f >= 20) & (f < 80)].sum() / (P[(f >= 200) & (f < 800)].sum() + 1e-30)
        res["rumble_db"] = round(float(10 * np.log10(lowr + 1e-30)), 1)
        res["voice_bandwidth"] = bandwidth(vocals[mask])
        res["speech_level_dbfs"] = round(float(np.median(lv[act])), 1)
    return res


def noise_floor(y):
    lv, _ = rms_db(y, 0.1, 0.05)
    return {"p5_dbfs": round(float(np.percentile(lv, 5)), 1), "p50_dbfs": round(float(np.percentile(lv, 50)), 1)}


def tonal_deviation(y, profile_ltas, cutoff=None, lo=50, hi=16000):
    """Per-band and RMS deviation of the long-term spectrum from a profile (both normalised at 0.5-2 kHz)."""
    cur = ltas(y)
    top = min(hi, (cutoff or hi) - 500)
    sel = (CENT >= lo) & (CENT <= top)
    d = cur - np.asarray(profile_ltas)
    return {"rms_db": round(float(np.sqrt(np.mean(d[sel] ** 2))), 2),
            "by_band": {int(round(c)): round(float(v), 1) for c, v in zip(CENT, d)
                        if c in CENT[sel] and int(round(c)) in (50, 79, 99, 125, 198, 315, 500, 794, 1000, 2000,
                                                                3175, 5040, 6350, 8000, 10079, 12699, 16000)}}
