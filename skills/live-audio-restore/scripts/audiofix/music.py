"""Music path: the strong remaster.

de-limit -> polarity repair -> de-ess / drum transients / width -> per-stem tonal match (per section) ->
stem balance (per section) -> per-stem dynamics -> bus glue -> overall tonal match (per section) ->
band extension -> bass to mono. Returns the pre-master mix; loudness and limiting happen in master.py.
"""
import json, os
import numpy as np

from . import SR, delimit, dsp, loudness, measure, profiles
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


def _stem_curve(stem_sec, mix_sec, ref_shape, core, p, keep_low, presence_db=None, fill_from=None):
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
    hi = np.where(CENT < 150, p["hi_low_db"], p["hi_db"])
    if presence_db:                                   # the vocal's presence band may move further (studio vocals)
        hi = np.where((CENT >= 1000) & (CENT <= 8000), np.maximum(hi, presence_db), hi)
    g = np.clip(g, p["lo_db"], hi)
    g[CENT < 40] = np.minimum(g[CENT < 40], 0)
    if keep_low:
        g[CENT < 120] = np.maximum(g[CENT < 120], 0)
    if fill_from:                                     # the top-band fill sets the level above its start: cut only
        g[CENT > fill_from - 500] = np.minimum(g[CENT > fill_from - 500], 0)
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


def snapshot(step, y, mask, prof_ltas, meas=None, cutoff=None):
    """One row of the step ledger: the music (mask) after a step. Loudness, 3 s crest, loudness range, mono-sum loss
    and the tone's RMS deviation from the reference (on the measured passage). A verify WARN is traced to the step
    that moved its measure, without control renders."""
    ym = y[mask] if mask is not None and mask.any() else y
    lu = loudness.integrated(ym)
    mid = ((ym[:, 0] + ym[:, 1]) / 2)[:, None]
    mono = loudness.integrated(np.repeat(mid, 2, axis=1))
    src = y[meas] if meas is not None and meas.sum() > 5 * SR else ym
    return {"step": step, "lufs": round(float(lu), 2), "crest3s_db": round(float(loudness.crest_3s(ym)), 2),
            "lra_lu": round(float(loudness.loudness_range(ym)), 2), "mono_loss_db": round(float(mono - lu), 2),
            "tone_rms_db": measure.tonal_deviation(src, prof_ltas, cutoff)["rms_db"]}


def _fmt(g):
    return " ".join(f"{f}:{np.interp(np.log(f), np.log(CENT), g):+.1f}" for f in (50, 80, 125, 250, 500, 1000, 2000,
                                                                                   4000, 6300, 10000, 12500))


def run(x, st, plan, music_mask, work, log=print):
    M = plan["music"]
    n = len(x)
    prof = profiles.load(plan["profiles"]["music"])
    secs = full_cover(plan.get("sections"), n / SR)
    W = dsp.ramp_weights(n, secs)
    # same-song reference: measure tone and balance on the passage the reference shows (like with like), and apply
    # the result to the whole song; otherwise per recording-chain section, measured over all of it
    ms = (M.get("match") or {}).get("measure_section")
    if ms:
        meas = np.zeros(n, bool)
        meas[int(ms[0] * SR):int(ms[1] * SR)] = True
        meas &= music_mask[:n]
        tsecs, TW = [[0.0, n / SR]], [np.ones(n, np.float32)]
    else:
        meas, tsecs, TW = music_mask, secs, W
    cutoff = plan["music"]["band_ext"]["cutoff_hz"] if M.get("band_ext") else None
    # with the top-band fill, everything above its start is set by the fill: no EQ step boosts there first (a stem EQ
    # that lifted Apollo's top left 16-18 kHz 1.5-3 dB over the reference, which the fill cannot take back)
    be0 = M.get("band_ext") or {}
    fill_from = (be0["cutoff_hz"] if be0.get("enabled") and be0.get("method", "fill") == "fill"
                 and profiles.hf_curve(prof) is not None else None)
    info = {"ledger": [], "balance": [], "overall_eq": []}
    mm = music_mask[:n]
    pl = np.asarray(prof["ltas"])

    def snap(step, y_now):
        row = snapshot(step, y_now, mm, pl, meas, cutoff)
        info["ledger"].append(row)
        log(f"[music]   after {step}: crest {row['crest3s_db']} dB, LRA {row['lra_lu']} LU, mono loss "
            f"{row['mono_loss_db']} dB, tone off by {row['tone_rms_db']} dB")

    snap("start", x)

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
        if gm is not None:
            snap("de-limit", y)
        if "mix" in M["antiphase"]["stems"]:
            y, a = dsp.fix_antiphase(y)
            log(f"[music] polarity repair on the mix: {a['treated_pct']:.0f} % of energy")
            snap("polarity", y)
    else:
        st = dict(st)
        res = x - sum(st.values())
        mixnow = lambda: sum(st.values()) + res                         # noqa: E731
        if gm is not None:
            st = {k: v * gm for k, v in st.items()}
            res = res * gm
            snap("de-limit", mixnow())
        # 2. polarity
        for s in M["antiphase"]["stems"]:
            if s in st:
                st[s], a = dsp.fix_antiphase(st[s])
                log(f"[music] polarity repair on {s}: {a['treated_pct']:.0f} % of its energy")
        if any(s in st for s in M["antiphase"]["stems"]):
            snap("polarity", mixnow())
        # 2b. room-miked images: each stem's width toward the reference stem's (vocal and drums back to the centre)
        wm = M.get("width_match", {})
        if wm.get("enabled"):
            sel = music_mask[:n]
            for s, tgt in (wm.get("targets") or {}).items():
                if tgt is None or s not in st:
                    continue
                mid = (st[s][:, 0] + st[s][:, 1]) / 2
                side = (st[s][:, 0] - st[s][:, 1]) / 2
                cur = 10 * np.log10(np.mean(side[sel].astype(np.float64) ** 2) /
                                    (np.mean(mid[sel].astype(np.float64) ** 2) + 1e-30) + 1e-12)
                if cur > tgt:
                    g = float(undb(max(tgt - cur, -wm.get("max_cut_db", 20.0))))
                    side = side * g
                    st[s] = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
                    log(f"[music] width {s}: side {cur:+.1f} -> {cur + 20 * np.log10(g):+.1f} dB re mid "
                        f"(reference {tgt:+.1f})")
            snap("width", mixnow())
        # 3. de-ess, transients, width
        if M["deess"]["enabled"]:
            st["vocals"], a = dsp.deess(st["vocals"], pct=M["deess"]["pct"], max_cut=M["deess"]["max_cut_db"])
            log(f"[music] de-ess: {a}")
            snap("de-ess", mixnow())
        if M["transients"]["enabled"]:
            st["drums"], a = dsp.transients(st["drums"], M["transients"]["amount"], M["transients"]["max_db"])
            log(f"[music] drum transients: {a}")
            snap("drum transients", mixnow())
        if M["widen"]["enabled"]:
            st["other"] = dsp.widen(st["other"], M["widen"]["side_db"])
            snap("widen", mixnow())
        # 4. per-stem tonal match, per section
        if M["stem_match"]["enabled"] and prof.get("stems"):
            mix = sum(st.values()) + res
            for s in st:
                core = profiles.CORE[s]
                pres = M["stem_match"].get("vocal_presence_db") if s == "vocals" else None
                curves = [_stem_curve(_sel(st[s], a, b, meas), _sel(mix, a, b, meas),
                                      np.asarray(prof["stems"][s]["shape"]), core, M["stem_match"],
                                      s in ("bass", "drums"), pres, fill_from) for a, b in tsecs]
                st[s] = dsp.apply_sectioned(st[s], [dsp.fir_from_curve(c) for c in curves], TW)
                for (a, b), c in zip(tsecs, curves):
                    log(f"[music] stem EQ {s:6s} {a:6.1f}-{b:6.1f}s  {_fmt(c)}")
                if s == "vocals":                       # the vocal chain's correction, reused for MCs (speech v2)
                    info["vocal_eq"] = {"cent": [round(float(c), 2) for c in CENT], "sections": tsecs,
                                        "curves": [[round(float(v), 2) for v in c] for c in curves],
                                        "mean": [round(float(v), 2) for v in np.mean(curves, axis=0)]}
                    with open(os.path.join(work, "vocal_eq.json"), "w") as fh:
                        json.dump(info["vocal_eq"], fh)
            snap("stem EQ", mixnow())
        # 5. balance, per section
        if M["balance"]["enabled"] and prof.get("stems"):
            gains = {s: np.zeros(n, np.float32) for s in st}
            for (a, b), w in zip(tsecs, TW):
                mixs = _sel(sum(st.values()) + res, a, b, meas)
                pm = np.mean(mixs.astype(np.float64) ** 2) + 1e-30
                cur_all = {s: 10 * np.log10(np.mean(_sel(st[s], a, b, meas).astype(np.float64) ** 2) / pm + 1e-30)
                           for s in st}
                absent_db = M["balance"].get("absent_db", -30.0)
                if cur_all.get("drums", 0.0) < absent_db:
                    # no rhythm section here: an acoustic or stripped-down arrangement, not the reference's full band
                    log(f"[music] balance {a:6.1f}-{b:6.1f}s: drums {cur_all['drums']:+.1f} dB re mix, absent: a "
                        f"stripped-down arrangement, the reference's stem ratios do not apply; nothing moved")
                    info["balance"].append({"section": [a, b], "drums_absent": round(cur_all["drums"], 1)})
                    continue
                for s in st:
                    cur = cur_all[s]
                    if cur < absent_db:                           # absent from this performance: nothing to balance
                        log(f"[music] balance {a:6.1f}-{b:6.1f}s {s:6s}: live {cur:+.1f} dB, absent: left alone")
                        continue
                    adj = float(np.clip((prof["stems"][s]["ratio_db"] - cur) * M["balance"]["fraction"],
                                        -M["balance"]["limit_db"], M["balance"]["limit_db"]))
                    gains[s] += w * adj
                    info["balance"].append({"section": [a, b], "stem": s, "live_db": round(cur, 1),
                                            "move_db": round(adj, 1), "at_limit": abs(adj) >= M["balance"]["limit_db"] - 0.05})
                    log(f"[music] balance {a:6.1f}-{b:6.1f}s {s:6s}: live {cur:+.1f} dB, reference "
                        f"{prof['stems'][s]['ratio_db']:+.1f} dB -> {adj:+.1f} dB")
            st = {s: v * undb(gains[s])[:, None].astype(np.float32) for s, v in st.items()}
            snap("balance", mixnow())
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
            snap("stem dynamics", mixnow())
        y = sum(st.values()) + res
        if D["enabled"]:
            r, at, rl, pc = D["bus"]
            y, a = dsp.compress(y, r, at, rl, pct=pc)
            log(f"[music] bus glue {r}:1: {a}")
            if r > 1:
                snap("bus glue", y)
    # 7. overall tonal match, per section
    if M["match"]["enabled"]:
        p = M["match"]
        curves = [dsp.match_curve(dsp.ltas(_sel(y, a, b, meas)), np.asarray(prof["ltas"]), p["alpha"],
                                  p["lo_db"], p["hi_db"], cutoff=cutoff) for a, b in tsecs]
        if fill_from:
            for c in curves:
                c[CENT > fill_from - 500] = np.minimum(c[CENT > fill_from - 500], 0)
        y = dsp.apply_sectioned(y, [dsp.fir_from_curve(c) for c in curves], TW)
        for (a, b), c in zip(tsecs, curves):
            log(f"[music] overall EQ {a:6.1f}-{b:6.1f}s  {_fmt(c)}")
            info["overall_eq"].append({"section": [a, b], "hi_db": p["hi_db"], "lo_db": p["lo_db"],
                                       "db": {f: round(float(np.interp(np.log(f), np.log(CENT), c)), 1)
                                              for f in (50, 80, 125, 250, 500, 1000, 2000, 4000, 6300, 10000, 12500)}})
        snap("overall EQ", y)
    # 8. band extension
    be = M.get("band_ext", {})
    curve = profiles.hf_curve(prof) if be.get("enabled") and be.get("method", "fill") == "fill" else None
    if curve is not None:
        y, a = dsp.hf_fill(y, be["cutoff_hz"], curve, below_db=be.get("below_ref_db", 0.0),
                           noise_mix=be.get("noise_mix", 0.5))
        info["band_ext"] = a
        log(f"[music] top rebuilt above {be['cutoff_hz']} Hz toward the reference, only what is missing: {a}")
        snap("band extension", y)
    elif be.get("enabled"):
        lo, hi = be["cutoff_hz"] + 300, min(be["cutoff_hz"] + 2500, 18000)
        tgt = profiles.hf_target(prof, lo, hi)
        if tgt is None:                                   # no profile data: continue the slope of the top octave
            tgt = dsp.band_rel(y, be["cutoff_hz"] - 2500, be["cutoff_hz"] - 300) - 6.0
        y, a = dsp.hf_extend(y, be["cutoff_hz"], tgt - be.get("below_ref_db", 0.0), noise_mix=be.get("noise_mix", 0.0))
        log(f"[music] band extension above {be['cutoff_hz']} Hz: {a}")
        snap("band extension", y)
    # 9. bass to mono
    if M.get("mono_bass_hz"):
        y = dsp.mono_bass(y, M["mono_bass_hz"])
        snap("mono bass", y)
    # 10. open recordings: the original's loudness contour back (quiet parts stay quiet)
    if M.get("macro", {}).get("enabled"):
        y, a = macro_restore(x, y, music_mask, M["macro"].get("max_db", 6.0))
        info["macro"] = a
        log(f"[music] loudness contour of the original restored: {a}")
        snap("loudness contour", y)
    return y.astype(np.float32), info
