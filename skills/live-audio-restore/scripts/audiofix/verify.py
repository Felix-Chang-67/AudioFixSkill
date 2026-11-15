"""Verification: the same measurements on input and output, pass/warn checks, a loudness-matched A/B file
and comparison plots. Nobody in this loop can listen except the user, so the numbers have to carry the claims."""
import json, os
import numpy as np

from . import SR, audio_io, dsp, loudness, measure, profiles, speech
from .analysis import cat, mask_of, regions


def _pick(score, starts, k, length):
    out = []
    for i in np.argsort(-score):
        t = starts[i]
        if all(abs(t - u) >= length + 2 for u in out):
            out.append(float(t))
        if len(out) >= k:
            break
    return sorted(out)


def ab_file(x, y, picks, path, length=7.0):
    parts = []
    fade = np.minimum(1, np.minimum(np.arange(int(length * SR)), np.arange(int(length * SR))[::-1]) / (0.05 * SR))[:, None]
    gap = lambda s: np.zeros((int(s * SR), 2), np.float32)
    for t in picks:
        a, b = int(t * SR), int((t + length) * SR)
        xo, yo = x[a:b], y[a:b]
        lx, ly = loudness.integrated(xo), loudness.integrated(yo)
        xo = xo * float(10 ** ((ly - lx) / 20))
        parts += [xo * fade, gap(0.6), yo * fade, gap(1.4)]
    ab = np.vstack(parts).astype(np.float32)
    pk = np.abs(ab).max()
    if pk > 0.89:
        ab *= 0.89 / pk
    audio_io.save_wav(path, ab, bits=16)


def run(work, plan, log=print, make_plots=True):
    D = json.load(open(os.path.join(work, "diagnosis.json")))
    x = audio_io.load(os.path.join(work, "input.wav"))
    y = audio_io.load(os.path.join(work, "restored.wav"))
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    segs = plan["segments"]
    mus, spc = regions(segs, "music"), regions(segs, "speech")
    V = {"input": {"loudness": loudness.summary(x), "stereo": measure.stereo(x), "bandwidth": measure.bandwidth(x),
                   "clipping": measure.clipping(x)},
         "output": {"loudness": loudness.summary(y), "stereo": measure.stereo(y), "bandwidth": measure.bandwidth(y),
                    "clipping": measure.clipping(y)}}
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    ceil = plan["master"]["ceiling_dbtp"]
    tp = V["output"]["loudness"]["true_peak_dbtp"]
    chk("true peak under the ceiling", tp <= ceil + 0.1, f"{tp} dBTP (ceiling {ceil})")
    chk("no new clipping", V["output"]["clipping"]["flat_runs_per_min"] <= max(5, V["input"]["clipping"]["flat_runs_per_min"]),
        f"{V['output']['clipping']['flat_runs_per_min']} flat runs/min")
    if mus:
        prof = profiles.load(plan["profiles"]["music"])
        xm, ym = cat(x, mus), cat(y, mus)
        cut = D["bandwidth"]["cutoff_hz"]
        ti = measure.tonal_deviation(xm, prof["ltas"], cut)
        to = measure.tonal_deviation(ym, prof["ltas"], cut)
        li, lo = loudness.summary(xm), loudness.summary(ym)
        si, so = measure.stereo(xm), measure.stereo(ym)
        V["music"] = {"input": {"loudness": li, "tonal": ti, "stereo": si}, "output": {"loudness": lo, "tonal": to, "stereo": so}}
        chk("music: closer to the reference tone", to["rms_db"] < ti["rms_db"], f"RMS deviation {ti['rms_db']} -> {to['rms_db']} dB")
        chk("music: dynamics not squashed further", lo["crest3s_db"] >= li["crest3s_db"] - 0.5,
            f"3 s crest {li['crest3s_db']} -> {lo['crest3s_db']} dB, PLR {li['plr_db']} -> {lo['plr_db']} dB")
        if so.get("kind") != "mono":
            chk("music: no anti-phase left in the mids", so.get("mid_corr_min", 1) >= -0.2,
                f"worst 4 s mid-band correlation {si.get('mid_corr_min')} -> {so.get('mid_corr_min')}")
            chk("music: mono compatible", so.get("mono_loss_db", 0) >= -1.5,
                f"mono-sum loss {si.get('mono_loss_db')} -> {so.get('mono_loss_db')} dB")
    if spc:
        sprof = profiles.load(plan["profiles"]["speech"])
        xs, ys = cat(x, spc), cat(y, spc)
        from .analysis import cat_frames
        mpath = os.path.join(work, "speech_mask.npy")
        if os.path.exists(mpath):
            act_full = np.load(mpath)
        else:
            act_full, _ = measure.speech_activity(x, spc)
        act, hop = cat_frames(act_full, spc, 0.01), 0.01
        mi, mo = measure.speech_metrics(xs, None, act, hop), measure.speech_metrics(ys, None, act, hop)
        if D.get("speech", {}).get("vad", {}).get("method") == "asr" and measure.asr_available():
            lang = D["speech"]["vad"].get("language")
            # Beam search with a seeded temperature fallback (reproducible; the fallback re-decodes looping windows),
            # every word kept and weighed by its probability: plain word counts move by several % between identical
            # unseeded runs, and Japanese word tokens merge and split. A loop that survives the fallback is dropped.
            kw = dict(language=lang, beam_size=5, strict=False, seed=0, max_compression=2.4)
            _, ai = measure.asr_activity(x, spc, **kw)
            _, ao = measure.asr_activity(y, spc, **kw)
            def talker(info):                       # words spoken by the talker = inside the analysis mask
                keep = np.array([p for t, p in zip(info.get("word_times", []), info.get("word_logprobs", []))
                                 if int(t / 0.01) < len(act_full) and act_full[int(t / 0.01)]])
                if not len(keep):
                    return 0, None, 0.0
                return len(keep), float(keep.mean()), float(np.exp(keep).sum())
            (wi, pi, wmi), (wo, po, wmo) = talker(ai), talker(ao)
            V["asr"] = {"input": {"talker_words": wi, "talker_logprob": pi, "word_mass": round(wmi, 1),
                                  "all_words": ai["words"], "loop_segments_dropped": ai["loop_segments_dropped"]},
                        "output": {"talker_words": wo, "talker_logprob": po, "word_mass": round(wmo, 1),
                                   "all_words": ao["words"], "loop_segments_dropped": ao["loop_segments_dropped"]}}
            if pi is not None and po is not None:
                # word mass = sum of word probabilities = expected number of correctly recognised words. Real speech
                # damage (neural denoisers) collapses it; run-to-run noise stays within a few %.
                chk("speech: the talker's words are recognised at least as well (nothing removed from the voice)",
                    wmo >= 0.9 * wmi,
                    f"Whisper on the mix, inside the talker mask: recognised-word mass {wmi:.1f} -> {wmo:.1f} "
                    f"({wi} words at mean log-prob {pi:.3f} -> {wo} at {po:.3f})")
        vb = D.get("speech", {}).get("metrics", {}).get("voice_bandwidth", {}).get("cutoff_hz")
        # the voice's tone, on the talker's frames, against the target this render aimed at
        mt = np.repeat(act_full, int(0.01 * SR))[:n]
        mt = np.concatenate([mt, np.zeros(n - len(mt), bool)]) & mask_of(n, spc)
        xt, yt = (x[mt], y[mt]) if mt.sum() > 3 * SR else (xs, ys)
        Sp = plan["speech"]
        if speech.mode_of(Sp) == "gentle":
            ref, label = np.asarray(sprof["ltas"]), "the speech spectrum"
        elif Sp["match"].get("target") == "song_vocal":
            ref, label = None, None                          # the songs' vocal EQ was copied: no spectral target
        else:
            t, kind = speech.voice_target(profiles.load(plan["profiles"]["music"]), sprof)
            ref = t - t[(dsp.CENT >= 500) & (dsp.CENT <= 2000)].mean()
            label = ("the studio vocal above 300 Hz, the speech spectrum below" if kind == "studio_vocal"
                     else "the speech spectrum")
        # When the voice is the mix (no stems, or separation lost it), the background owns part of the spectrum and
        # denoising it moves the "voice" spectrum too: the voice's tone cannot be judged on its own, numbers only.
        voice_is_mix = Sp["source"]["value"] == "mix" or not D.get("stems_dir")
        ti = measure.tonal_deviation(xt, ref if ref is not None else sprof["ltas"], vb, lo=100)
        to = measure.tonal_deviation(yt, ref if ref is not None else sprof["ltas"], vb, lo=100)
        V["speech"] = {"input": {"metrics": mi, "tonal": ti, "loudness": loudness.summary(xs)},
                       "output": {"metrics": mo, "tonal": to, "loudness": loudness.summary(ys)},
                       "tone_reference": label or "speech spectrum (for information; the songs' vocal EQ was applied)"}
        if plan["speech"]["crowd"]["enabled"] and "gap_minus_speech_db" in mi:
            chk("speech: crowd no longer competes with the talk",
                mo.get("gap_minus_speech_db", 99) <= min(mi["gap_minus_speech_db"] - 3, 0),
                f"gaps re speech, median {mi['gap_minus_speech_db']} -> {mo.get('gap_minus_speech_db')} dB, loudest 10 % "
                f"{mi.get('gap_p90_minus_speech_db')} -> {mo.get('gap_p90_minus_speech_db')} dB")
        if ref is not None and not voice_is_mix:
            chk(f"speech: voice closer to its target ({label})", to["rms_db"] <= ti["rms_db"] + 0.2,
                f"RMS deviation on the talker's frames {ti['rms_db']} -> {to['rms_db']} dB")
    V["checks"] = checks

    # A/B: music passages with the most voice (or level), spoken passages with the most talk
    picks = []
    hopw = int(SR)
    starts = np.arange(0, max(1, n - 8 * SR), hopw) / SR
    lvl = np.array([np.mean(x[int(t * SR):int(t * SR) + 7 * SR].astype(np.float64) ** 2) for t in starts])
    for regs, k in ((mus, 2 if spc else 4), (spc, 2 if mus else 4)):
        if not regs:
            continue
        inside = np.array([any(a <= t and t + 7 <= b for a, b in regs) for t in starts])
        if inside.any():
            picks += _pick(np.where(inside, lvl, -1), starts, k, 7.0)
    picks = sorted(set(picks))
    if picks:
        ab_file(x, y, picks, os.path.join(work, "AB.wav"))
        V["ab_times_s"] = picks
    if make_plots:
        try:
            plots(x, y, plan, picks, os.path.join(work, "compare.png"))
        except Exception as e:                                   # noqa: BLE001
            log(f"[verify] plots skipped: {e}")
    with open(os.path.join(work, "verification.json"), "w") as fh:
        json.dump(V, fh, ensure_ascii=False, indent=1, default=float)
    ok = sum(c["pass"] for c in checks)
    log(f"[verify] {ok}/{len(checks)} checks passed")
    for c in checks:
        log(f"   {'PASS' if c['pass'] else 'WARN'}  {c['check']}: {c['detail']}")
    return V


def plots(x, y, plan, picks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    segs = plan["segments"]
    rows = 1 + min(2, len(picks))
    fig, ax = plt.subplots(rows, 2, figsize=(15, 4 * rows), squeeze=False)
    for col, (label, regs, pname) in enumerate((("music", regions(segs, "music"), plan["profiles"]["music"]),
                                                ("speech", regions(segs, "speech"), plan["profiles"]["speech"]))):
        a = ax[0, col]
        if regs:
            prof = profiles.load(pname)
            a.semilogx(dsp.CENT, prof["ltas"], "k", lw=2, label=f"profile {pname}")
            a.semilogx(dsp.CENT, dsp.ltas(cat(x, regs)), "C0", label="input")
            a.semilogx(dsp.CENT, dsp.ltas(cat(y, regs)), "C3", label="restored")
            a.legend(); a.grid(alpha=.3, which="both"); a.set_xlim(30, 20000)
        a.set_title(f"{label}: long-term spectrum (dB re 0.5-2 kHz)")
    for r, t in enumerate(picks[:rows - 1], start=1):
        for col, (name, z) in enumerate((("input", x), ("restored", y))):
            seg = z[int(t * SR):int((t + 7) * SR)].mean(axis=1)
            f, tt, Z = stft(seg, SR, nperseg=2048, noverlap=1536)
            ax[r, col].pcolormesh(tt + t, f / 1000, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-110, vmax=-20,
                                  cmap="magma", shading="auto")
            ax[r, col].set_ylim(0, 21); ax[r, col].set_title(f"{name} {t:.0f}-{t + 7:.0f} s"); ax[r, col].set_ylabel("kHz")
    plt.tight_layout(); plt.savefig(path, dpi=60); plt.close(fig)
