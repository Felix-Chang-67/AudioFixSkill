"""Music path: the strong remaster.

de-limit -> polarity repair -> de-ess / drum transients / width -> per-stem tonal match (per section) ->
stem balance (per section) -> per-stem dynamics -> bus glue -> overall tonal match (per section) ->
band extension -> bass to mono. Returns the pre-master mix; loudness and limiting happen in master.py.
"""
import json, os
import numpy as np

from . import SR, delimit, dsp, loudness, profiles
from .dsp import CENT, undb


def full_cover(sections, total):
    """Sections that tile the whole file (gaps split at their middle), so the weights sum to 1 everywhere."""
    if not sections:
        return [[0.0, total]]
    s = [list(x) for x in sorted(sections)]
    s[0][0], s[-1][1] = 0.0, total
    for i in range(len(s) - 1):
        if s[i][1] < s[i + 1][0]:
            m = (s[i][1] + s[i + 1][0]) / 2
            s[i][1], s[i + 1][0] = m, m
    return s


def _sel(arr, a, b, mask):
    i0, i1 = int(a * SR), int(b * SR)
    m = mask[i0:i1]
    seg = arr[i0:i1]
    return seg[m] if m.sum() > 5 * SR else seg


def _stem_curve(stem_sec, mix_sec, ref_shape, core, p, keep_low):
    f, P = dsp.psd(stem_sec)
    _, Mx = dsp.psd(mix_sec)
    Bs, Bm = dsp.bands(f, P), dsp.bands(f, Mx)
    sel = (CENT >= core[0]) & (CENT <= core[1])
    cur = 10 * np.log10(Bs + 1e-30)
    cur -= cur[sel].mean()
    from scipy.ndimage import uniform_filter1d
    d = uniform_filter1d(ref_shape - cur, 3, mode="nearest")
    conf = np.clip((Bs / (Bm + 1e-30) - 0.05) / 0.25, 0, 1)          # boost only where the stem is really present
    g = d * p["alpha"]
    g = np.where(g > 0, g * conf, g)
    g = np.clip(g, p["lo_db"], np.where(CENT < 150, p["hi_low_db"], p["hi_db"]))
    g[CENT < 40] = np.minimum(g[CENT < 40], 0)
    if keep_low:
        g[CENT < 120] = np.maximum(g[CENT < 120], 0)
    return g


def macro_restore(x, y, mask, max_db=6.0, hop_s=0.5, win_s=3.0, smooth_s=4.0):
    """Put the original's slow loudness contour back on the processed music: K-weighted 3 s level of input and
    output, each relative to its own median over the music, difference smoothed over 4 s and applied as a slow gain
    (within +-max_db). Tone EQ, stem balance and compression all change how loud a quiet intro is next to a chorus;
    this undoes that at the scale of seconds without touching the hits or the tone."""
    from scipy.ndimage import uniform_filter1d
    h = int(hop_s * SR)
    n = min(len(x), len(y)) // h
    if n < 20:
        return y, {"applied": False}

    def env(z):
        k = loudness.kweight(z[:n * h])
        p = (k.astype(np.float64) ** 2).sum(axis=1).reshape(n, h).mean(axis=1)
        return 10 * np.log10(uniform_filter1d(p, max(1, int(win_s / hop_s))) + 1e-12)

    ex, ey = env(x), env(y)
    mf = mask[:n * h].reshape(n, h).mean(axis=1) > 0.5
    if mf.sum() < 10:
        return y, {"applied": False}
    valid = mf & (ex > np.median(ex[mf]) - 30)
    d = np.where(valid, (ex - np.median(ex[valid])) - (ey - np.median(ey[valid])), 0.0)
    d = np.clip(uniform_filter1d(d, max(1, int(smooth_s / hop_s))), -max_db, max_db)
    g = np.interp(np.arange(len(y)), (np.arange(n) + 0.5) * h, d)
    return (y * undb(g)[:, None]).astype(np.float32), {"mean_abs_db": round(float(np.mean(np.abs(d[valid]))), 2),
                                                       "max_db": round(float(np.max(np.abs(d[valid]))), 2)}


def _fmt(g):
    return " ".join(f"{f}:{np.interp(np.log(f), np.log(CENT), g):+.1f}" for f in (50, 80, 125, 250, 500, 1000, 2000,
                                                                                   4000, 6300, 10000, 12500))


def run(x, st, plan, music_mask, work, log=print):
    M = plan["music"]
    n = len(x)
    prof = profiles.load(plan["profiles"]["music"])
    secs = full_cover(plan.get("sections"), n / SR)
    W = dsp.ramp_weights(n, secs)
    cutoff = plan["music"]["band_ext"]["cutoff_hz"] if M.get("band_ext") else None
    info = {}

    # 1. de-limit
    gm = None
    if M["delimit"]["enabled"]:
        cache = os.path.join(work, "delimit_gain.npy")
        if os.path.exists(cache):
            g = np.load(cache)[:n]
        else:
            log("[music] de-limiter inference ...")
            g = delimit.gain_curve(x)
            np.save(cache, g)
        gm = delimit.modulation(g, M["delimit"]["depth"])[:, None]
        mod = 20 * np.log10(gm[:, 0])
        info["delimit_mod_p1_p99_db"] = [round(float(np.percentile(mod, 1)), 2), round(float(np.percentile(mod, 99)), 2)]
        log(f"[music] de-limit depth {M['delimit']['depth']}: modulation {info['delimit_mod_p1_p99_db']} dB (p1..p99)")

    if st is None:                                                   # ---------------- mix-only fallback
        y = x if gm is None else x * gm
        if "mix" in M["antiphase"]["stems"]:
            y, a = dsp.fix_antiphase(y)
            log(f"[music] polarity repair on the mix: {a['treated_pct']:.0f} % of energy")
    else:
        st = dict(st)
        res = x - sum(st.values())
        if gm is not None:
            st = {k: v * gm for k, v in st.items()}
            res = res * gm
        # 2. polarity
        for s in M["antiphase"]["stems"]:
            if s in st:
                st[s], a = dsp.fix_antiphase(st[s])
                log(f"[music] polarity repair on {s}: {a['treated_pct']:.0f} % of its energy")
        # 3. de-ess, transients, width
        if M["deess"]["enabled"]:
            st["vocals"], a = dsp.deess(st["vocals"], pct=M["deess"]["pct"], max_cut=M["deess"]["max_cut_db"])
            log(f"[music] de-ess: {a}")
        if M["transients"]["enabled"]:
            st["drums"], a = dsp.transients(st["drums"], M["transients"]["amount"], M["transients"]["max_db"])
            log(f"[music] drum transients: {a}")
        if M["widen"]["enabled"]:
            st["other"] = dsp.widen(st["other"], M["widen"]["side_db"])
        # 4. per-stem tonal match, per section
        if M["stem_match"]["enabled"] and prof.get("stems"):
            mix = sum(st.values()) + res
            for s in st:
                core = profiles.CORE[s]
                curves = [_stem_curve(_sel(st[s], a, b, music_mask), _sel(mix, a, b, music_mask),
                                      np.asarray(prof["stems"][s]["shape"]), core, M["stem_match"],
                                      s in ("bass", "drums")) for a, b in secs]
                st[s] = dsp.apply_sectioned(st[s], [dsp.fir_from_curve(c) for c in curves], W)
                for (a, b), c in zip(secs, curves):
                    log(f"[music] stem EQ {s:6s} {a:6.1f}-{b:6.1f}s  {_fmt(c)}")
                if s == "vocals":                       # the vocal chain's correction, reused for MCs (speech v2)
                    info["vocal_eq"] = {"cent": [round(float(c), 2) for c in CENT], "sections": secs,
                                        "curves": [[round(float(v), 2) for v in c] for c in curves],
                                        "mean": [round(float(v), 2) for v in np.mean(curves, axis=0)]}
                    with open(os.path.join(work, "vocal_eq.json"), "w") as fh:
                        json.dump(info["vocal_eq"], fh)
        # 5. balance, per section
        if M["balance"]["enabled"] and prof.get("stems"):
            gains = {s: np.zeros(n, np.float32) for s in st}
            for (a, b), w in zip(secs, W):
                mixs = _sel(sum(st.values()) + res, a, b, music_mask)
                pm = np.mean(mixs.astype(np.float64) ** 2) + 1e-30
                for s in st:
                    cur = 10 * np.log10(np.mean(_sel(st[s], a, b, music_mask).astype(np.float64) ** 2) / pm + 1e-30)
                    adj = float(np.clip((prof["stems"][s]["ratio_db"] - cur) * M["balance"]["fraction"],
                                        -M["balance"]["limit_db"], M["balance"]["limit_db"]))
                    gains[s] += w * adj
                    log(f"[music] balance {a:6.1f}-{b:6.1f}s {s:6s}: live {cur:+.1f} dB, reference "
                        f"{prof['stems'][s]['ratio_db']:+.1f} dB -> {adj:+.1f} dB")
            st = {s: v * undb(gains[s])[:, None].astype(np.float32) for s, v in st.items()}
        # 6. dynamics per stem
        D = M["dynamics"]
        if D["enabled"]:
            dm = st["drums"][music_mask] if music_mask.any() else st["drums"]
            dl = loudness.integrated(dm)
            st["drums"], a = dsp.limit(st["drums"], ceiling_db=dl + D["drum_ceiling_over_lufs"], look_ms=1.0,
                                       release_ms=50)
            log(f"[music] drum peaks held at {dl + D['drum_ceiling_over_lufs']:+.1f} dBTP: {a}")
            for s in ("bass", "vocals", "other"):
                r, at, rl, pc = D[s]
                st[s], a = dsp.compress(st[s], r, at, rl, pct=pc)
                log(f"[music] compress {s}: {r}:1 {a}")
        y = sum(st.values()) + res
        if D["enabled"]:
            r, at, rl, pc = D["bus"]
            y, a = dsp.compress(y, r, at, rl, pct=pc)
            log(f"[music] bus glue {r}:1: {a}")
    # 7. overall tonal match, per section
    if M["match"]["enabled"]:
        p = M["match"]
        curves = [dsp.match_curve(dsp.ltas(_sel(y, a, b, music_mask)), np.asarray(prof["ltas"]), p["alpha"],
                                  p["lo_db"], p["hi_db"], cutoff=cutoff) for a, b in secs]
        y = dsp.apply_sectioned(y, [dsp.fir_from_curve(c) for c in curves], W)
        for (a, b), c in zip(secs, curves):
            log(f"[music] overall EQ {a:6.1f}-{b:6.1f}s  {_fmt(c)}")
    # 8. band extension
    be = M.get("band_ext", {})
    if be.get("enabled"):
        lo, hi = be["cutoff_hz"] + 300, min(be["cutoff_hz"] + 2500, 18000)
        tgt = profiles.hf_target(prof, lo, hi)
        if tgt is None:                                   # no profile data: continue the slope of the top octave
            tgt = dsp.band_rel(y, be["cutoff_hz"] - 2500, be["cutoff_hz"] - 300) - 6.0
        y, a = dsp.hf_extend(y, be["cutoff_hz"], tgt - be.get("below_ref_db", 0.0))
        log(f"[music] band extension above {be['cutoff_hz']} Hz: {a}")
    # 9. bass to mono
    if M.get("mono_bass_hz"):
        y = dsp.mono_bass(y, M["mono_bass_hz"])
    # 10. open recordings: the original's loudness contour back (quiet parts stay quiet)
    if M.get("macro", {}).get("enabled"):
        y, a = macro_restore(x, y, music_mask, M["macro"].get("max_db", 6.0))
        info["macro"] = a
        log(f"[music] loudness contour of the original restored: {a}")
    return y.astype(np.float32), info
