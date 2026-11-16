"""Mastering: join the music and speech renders with short crossfades, set loudness, limit true peaks.

Music loudness is the loudest level at which the lookahead limiter would work above 1 dB at most over1_max_pct
of the time and above 2 dB at most over2_max_pct (perceptual criterion: the limiter stays out of the way).
Spoken parts of a concert sit speech_offset_db under the music; a speech-only file goes to speech_only_lufs."""
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import SR, dsp, loudness
from .analysis import cat, mask_of, regions
from .dsp import undb


def class_weights(n, segs, fade_s=1.0):
    m = mask_of(n, regions(segs, "music")).astype(np.float32)
    if 0 < m.mean() < 1:
        m = uniform_filter1d(m, int(fade_s * SR)).astype(np.float32)
    return m, 1.0 - m


def music_target(ym, lu0, M, log):
    pk = dsp.isp_peaks(ym)
    lo, hi = M["search_lufs"]
    t = lo
    while t >= hi - 1e-9:
        g = dsp.limiter_gain(pk, M["ceiling_dbtp"], pre_gain_db=t - lu0)
        gr = -dsp.db(np.maximum(g, 1e-6))
        o1, o2 = float(np.mean(gr > 1) * 100), float(np.mean(gr > 2) * 100)
        log(f"[master] try {t:.1f} LUFS: limiter above 1 dB {o1:.1f} %, above 2 dB {o2:.1f} %")
        if o1 <= M["over1_max_pct"] and o2 <= M["over2_max_pct"]:
            mc = M.get("min_crest_db")
            if mc is None:
                return t
            # open recordings: also keep the 3 s crest within 0.5 dB of the original (what verify checks)
            c = loudness.crest_3s(ym * float(undb(t - lu0)) * np.reshape(g, (-1, 1)))
            log(f"[master]   3 s crest at {t:.1f} LUFS: {c:.2f} dB (keep >= {mc:.2f})")
            if c >= mc:
                return t
        t -= 0.5
    return hi


def finalize(y_music, y_speech, plan, log=print):
    M = plan["master"]
    segs = plan["segments"]
    n = len(y_music if y_music is not None else y_speech)
    wm, ws = class_weights(n, segs)
    mus, spc = regions(segs, "music"), regions(segs, "speech")
    out = np.zeros((n, 2), np.float32)
    rep = {}
    t_m = None
    if mus and y_music is not None:
        ym = cat(y_music, mus)
        lu0 = loudness.integrated(ym)
        t_m = music_target(ym, lu0, M, log)
        out += y_music * float(undb(t_m - lu0)) * wm[:, None]
        rep["music_target_lufs"] = t_m
    if spc and y_speech is not None:
        ys = cat(y_speech, spc)
        lu_s = loudness.integrated(ys)
        t_s = (t_m - M["speech_offset_db"]) if t_m is not None else M["speech_only_lufs"]
        out += y_speech * float(undb(t_s - lu_s)) * ws[:, None]
        rep["speech_target_lufs"] = t_s
    y, st = dsp.limit(out, M["ceiling_dbtp"])
    if t_m is not None:                                   # one correction pass: the limiter shaves a little loudness
        got = loudness.integrated(cat(y, mus))
        if abs(t_m - got) > 0.1:
            y, st = dsp.limit(out * float(undb(t_m - got)), M["ceiling_dbtp"])
    elif spc:
        got = loudness.integrated(cat(y, spc))
        if abs(rep["speech_target_lufs"] - got) > 0.1:
            y, st = dsp.limit(out * float(undb(rep["speech_target_lufs"] - got)), M["ceiling_dbtp"])
    tp = loudness.true_peak(y)
    if tp > M["ceiling_dbtp"] + 0.05:                      # belt and braces after resampling ripple
        y = (y * float(undb(M["ceiling_dbtp"] - 0.05 - tp))).astype(np.float32)
    rep.update(limiter=st, final=loudness.summary(y))
    log(f"[master] limiter {st}; final {rep['final']}")
    return y, rep
