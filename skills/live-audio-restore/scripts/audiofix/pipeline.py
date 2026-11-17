"""Render a plan: raw-mix repairs, music path, speech path, mastering, outputs."""
import json, os, subprocess, time
import numpy as np

from . import SR, audio_io, dsp, master, music, speech, stems as S
from .analysis import mask_of, regions


def _birdies(y, R, log=None):
    b = R.get("birdies") or {}
    if not b.get("enabled"):
        return y
    y = dsp.notches(y, b["hz"], b.get("bw_hz", 15.0))
    if log:
        log(f"[repair] notched {len(b['hz'])} codec tone(s), steady or on and off: {b['hz']} Hz")
    return y


def _sbr(y, R, log=None, cut=None):
    """Cut the tones band replication copied up; the cut is measured on the mix and applied alike to every stem."""
    s = R.get("sbr_copies") or {}
    if not s.get("enabled") or not s.get("shifts"):
        return y, None
    if cut is None:
        cut = dsp.sbr_cut(y.mean(axis=1), s["shifts"], min_db=s.get("min_db", 6.0),
                          max_cut_db=s.get("max_cut_db", 30.0))
        if log:
            c = cut[1]
            on = c > 3
            log(f"[repair] cut the tones band replication copied up ({', '.join(str(v['shift_hz']) for v in s['shifts'])}"
                f" Hz shift): in {np.mean(on.any(axis=0)) * 100:.0f} % of the frames, {np.mean(on) * 100:.0f} % of the "
                f"band's bins, by {c[on].mean() if on.any() else 0:.1f} dB on average")
    return dsp.apply_cut(y, *cut), cut


def _repairs(x, R, log):
    y = dsp.highpass(x, R["highpass_hz"]["value"], order=2)
    if R.get("dehum", {}).get("enabled"):
        for k in R["dehum"]["harmonics"]:
            y = dsp.notch(y, R["dehum"]["mains_hz"] * k, q=35.0)
        log(f"[repair] notched {R['dehum']['mains_hz']} Hz x{R['dehum']['harmonics']}")
    y = _birdies(y, R, log)
    y, R_cut = _sbr(y, R, log)
    if R.get("declip", {}).get("enabled") and audio_io.have_ffmpeg():
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                            "-af", "adeclip", "-f", "f32le", "-"], input=np.ascontiguousarray(y).tobytes(),
                           capture_output=True, check=True)
        z = np.frombuffer(p.stdout, np.float32).reshape(-1, 2)[:len(y)]
        y = np.ascontiguousarray(z) if len(z) == len(y) else y
        log("[repair] declipped (ffmpeg adeclip)")
    return y.astype(np.float32), R_cut


def restore(work, plan, log=print):
    t0 = time.time()
    D = json.load(open(os.path.join(work, "diagnosis.json")))
    rep_wav = os.path.join(work, "input_repaired.wav")
    use_rep = D.get("codec_repair", {}).get("applied") and plan["repair"].get("codec", {}).get("enabled", True) \
        and os.path.exists(rep_wav)
    x = audio_io.load(rep_wav if use_rep else os.path.join(work, "input.wav"))
    if use_rep:
        log("[repair] using the codec-repaired input (Apollo)")
    n = len(x)
    x, sbr_cut = _repairs(x, plan["repair"], log)
    st = None
    if D.get("stems_dir") and (plan.get("music") or (plan.get("speech") and plan["speech"]["source"]["value"] == "stems")):
        st = S.load(D["stems_dir"], n)
        st = {k: dsp.highpass(v, plan["repair"]["highpass_hz"]["value"], order=2) for k, v in st.items()}
        if plan["repair"].get("dehum", {}).get("enabled"):
            for k in list(st):
                for h in plan["repair"]["dehum"]["harmonics"]:
                    st[k] = dsp.notch(st[k], plan["repair"]["dehum"]["mains_hz"] * h, q=35.0)
        st = {k: _sbr(_birdies(v, plan["repair"]), plan["repair"], cut=sbr_cut)[0] for k, v in st.items()}
    segs = plan["segments"]
    mus, spc = regions(segs, "music"), regions(segs, "speech")
    y_m = y_s = None
    info = {}
    if plan.get("music") and mus:
        log("[restore] music path ...")
        y_m, info["music"] = music.run(x, st, plan, mask_of(n, mus), work, log)
    if plan.get("speech") and spc:
        log("[restore] speech path ...")
        mpath = os.path.join(work, "speech_mask.npy")
        act = np.load(mpath) if os.path.exists(mpath) else None
        if act is None:
            from . import measure
            act, _ = measure.speech_activity(st["vocals"] if st is not None else x, spc)
        y_s, _ = speech.run(x, st, plan, mask_of(n, spc), act, 0.01, log, work=work,
                            chain=info.get("music", {}).get("vocal_eq"))
    y, rep = master.finalize(y_m, y_s, plan, log)
    if info.get("music", {}).get("ledger") is not None:
        from . import profiles
        mi = info["music"]
        ms = (plan["music"].get("match") or {}).get("measure_section")
        mm = mask_of(n, mus)
        meas = (mm & mask_of(n, [ms])) if ms else mm
        row = music.snapshot("master", y, mm, profiles.load(plan["profiles"]["music"])["ltas"], meas,
                             (plan["music"].get("band_ext") or {}).get("cutoff_hz"))
        mi["ledger"].append(row)
        log(f"[music]   after master: crest {row['crest3s_db']} dB, LRA {row['lra_lu']} LU, loudness {row['lufs']} LUFS")
        with open(os.path.join(work, "steps.json"), "w") as fh:
            json.dump({k: mi.get(k) for k in ("ledger", "balance", "overall_eq", "dynamics", "band_ext")}, fh,
                      ensure_ascii=False, indent=1, default=float)
    out = os.path.join(work, "restored.wav")
    audio_io.save_wav(out, y, bits=24)
    audio_io.encode_aac(out, os.path.join(work, "restored.m4a"))
    rep["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(work, "master.json"), "w") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=float)
    return out, rep
