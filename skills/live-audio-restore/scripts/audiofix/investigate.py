"""Investigate a listener's complaint the way an engineer would, so that any assistant does it, not only one that
already knows how.

1. The listener's words become hypotheses (CATEGORIES -> probes).
2. Each hypothesis is measured where it could arise: the original, the codec-repaired input, the render and the step
   ledger. The evidence is ranked.
3. One variant per likely cause is rendered, each changing one setting, and a listening file plays the current render
   and every variant, passage by passage. investigate.md says what each variant changed and what it measured.
4. The listener chooses; the chosen settings are applied with `restore --set ... --why "<their choice>"`.

The assistant that only mapped words to a knob changed the wrong thing, or nothing, until it was told the cause. The
one that investigated found, for a "digital feel", 15 steady whistles that the original had 4 of."""
import json, os, subprocess, sys

import numpy as np

from . import SR, audio_io, loudness, measure, profiles

# words (any language the users wrote in) -> complaint categories
CATEGORIES = {
    "digital": ("digital, fake, metallic or whistling highs",
                ["数码", "电子味", "假", "金属", "啸", "哨", "嘶", "digital", "fake", "metallic", "whistl", "fizz",
                 "artificial", "synthetic", "sizzl", "デジタル"]),
    "harsh": ("harsh, sibilant or piercing", ["刺", "尖", "齿音", "s音", "harsh", "sibil", "piercing", "shrill", "耳が痛"]),
    "scattered": ("scattered, unfocused, phasey or hollow",
                  ["散", "不聚焦", "发飘", "飘", "空洞", "相位", "scatter", "unfocus", "phasey", "hollow", "diffuse"]),
    "processed": ("processed, unnatural, watery or pumping",
                  ["处理痕迹", "不自然", "不纯净", "水声", "抽", "processed", "artefact", "artifact", "watery", "swirl",
                   "unnatural", "pumping", "不干净"]),
    "muddy": ("muddy, boxy or muffled", ["浑", "糊", "闷", "muddy", "boxy", "muffled"]),
    "thin": ("thin, no low end", ["薄", "没低频", "低频不够", "thin", "no bass", "weak low"]),
    "boomy": ("boomy, too much low end", ["轰", "低频太多", "boom", "too much bass"]),
    "squashed": ("squashed, flat, no punch", ["扁", "压缩", "没动态", "没冲击", "squash", "flat", "no punch", "lifeless"]),
}
VOCAL_WORDS = ["人声", "vocal", "voice", "唱", "歌声", "声音"]


def categories(text):
    t = (text or "").lower()
    cats = [k for k, (_, words) in CATEGORIES.items() if any(w.lower() in t for w in words)]
    return cats or list(CATEGORIES), any(w.lower() in t for w in VOCAL_WORDS)


def _load(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _rel_spectrum(y):
    from scipy.signal import welch
    f, P = welch(y.mean(axis=1), SR, nperseg=8192)
    ref = P[(f >= 500) & (f <= 2000)].mean() + 1e-30                   # the profile's own reference band
    return f, 10 * np.log10(P / ref + 1e-30)


def _band(f, d, lo, hi):
    s = (f >= lo) & (f < hi)
    return float(np.mean(d[s])) if s.any() else float("nan")


class Context:
    def __init__(self, work):
        self.work = work
        self.D = _load(os.path.join(work, "diagnosis.json"))
        self.P = _load(os.path.join(work, "plan_used.json")) or _load(os.path.join(work, "plan.json"))
        self.V = _load(os.path.join(work, "verification.json"))
        self.S = _load(os.path.join(work, "steps.json"))
        self.M = _load(os.path.join(work, "master.json"))
        self.orig = audio_io.load(os.path.join(work, "input.wav"))
        rep = os.path.join(work, "input_repaired.wav")
        self.rep = audio_io.load(rep) if os.path.exists(rep) else None
        self.out = audio_io.load(os.path.join(work, "restored.wav"))
        self.prof = profiles.load((self.P.get("profiles") or {}).get("music", "jrock_band"))
        self.curve = profiles.hf_curve(self.prof)
        self.music = self.P.get("music") or {}
        self.cut = (self.D.get("bandwidth") or {}).get("cutoff_hz")


# ------------------------------------------------------------------ probes: each returns a finding or None
def _whistles(y):
    """Steady lines (long-term spectrum) and on-and-off ones (short whistles piling up at one frequency)."""
    steady = measure.tonal_lines(y)
    onoff = [w for w in measure.whistle_lines(y) if all(abs(w["hz"] - s["hz"]) > 12 for s in steady)]
    return steady, onoff


def p_whistles(c):
    steady, onoff = _whistles(c.out)
    b_steady, b_onoff = _whistles(c.orig)
    copies = measure.sbr_copies(c.out, c.cut)            # tones band replication copied up, still there
    b_copies = measure.sbr_copies(c.orig, c.cut)
    hyp = "whistles (narrow tones) in the highs: steady, on and off, or notes copied up by the codec"
    before = (f"the original had {len(b_steady)} steady and {len(b_onoff)} on and off"
              + (f", and its top band was the low band shifted up {b_copies[0]['shift_hz']} Hz (correlation "
                 f"{b_copies[0]['corr']})" if b_copies else ""))
    if not steady and not onoff and not copies:
        return {"id": "whistles", "score": 0.0, "hypothesis": hyp, "evidence": f"none in the render ({before})"}
    lines = steady + onoff
    known = set((c.P.get("repair") or {}).get("birdies", {}).get("hz") or [])
    hz = sorted(known | {line["hz"] for line in lines})
    where = ", ".join(f"{line['hz'] / 1000:.2f}" for line in lines)
    secs = sum(w["total_s"] for w in onoff)
    # a steady line counts from 6 dB over its neighbours in the long-term spectrum; an on-and-off one is measured
    # in short frames, where 18 dB is the floor
    score = max([np.clip((s["excess_db"] - 6) / 10, 0.2, 1.0) for s in steady]
                + [np.clip(0.4 + w["total_s"] / 10, 0.4, 1.0) for w in onoff]
                + [np.clip(0.4 + s["corr"], 0.4, 1.0) for s in copies])
    ev = []
    if lines:
        ev.append(f"{len(steady)} steady and {len(onoff)} on and off in the render"
                  + (f" (whistling {secs:.0f} s in total)" if onoff else "") + f", at {where} kHz")
    for s in copies:
        ev.append(f"its {s['band_hz'][0]}-{s['band_hz'][1]} Hz band is still the low band shifted up {s['shift_hz']} Hz "
                  f"(correlation {s['corr']})")
    var = {}
    if lines:
        var.update({"repair.birdies.enabled": True, "repair.birdies.hz": hz})
    if copies:
        var.update({"repair.sbr_copies.enabled": True,
                    "repair.sbr_copies.shifts": [{"shift_hz": s["shift_hz"], "band_hz": s["band_hz"]} for s in copies]})
    return {"id": "whistles", "score": float(score), "hypothesis": hyp,
            "evidence": "; ".join(ev) + f" ({before})", "variant": var,
            "label": "notch every whistle left" + (" and cut the copied tones" if copies else "")}


def p_top(c):
    be = c.music.get("band_ext") or {}
    if not be.get("enabled") or c.curve is None:
        return None
    f, d = _rel_spectrum(c.out)
    start = be.get("cutoff_hz", 16000)
    devs = [(_band(f, d, lo, lo + 1000) - c.curve(lo + 500)) for lo in range(int(max(start, 11000)), 19000, 1000)]
    mean_dev = float(np.nanmean(devs)) if devs else 0.0
    synth = max(0.0, (19500 - start) / 1000)
    score = float(np.clip(0.25 + mean_dev / 4 + (0.25 if start < 12500 else 0.0), 0.0, 1.0))
    below = float(be.get("below_ref_db", 0.0))
    return {"id": "top", "score": score, "hypothesis": "the rebuilt top is too present for this recording",
            "evidence": f"{synth:.1f} kHz of the top is rebuilt (from {start} Hz), at {mean_dev:+.1f} dB re the "
                        f"reference's level; listeners heard a top at the studio level as 'digital, the vocal "
                        f"scattered' on live recordings, and one 4 dB darker as clean",
            "variant": {"music.band_ext.below_ref_db": below + 4}, "label": "rebuilt top 4 dB darker",
            "bracket": {"music.band_ext.enabled": False}, "bracket_label": "no rebuilt top at all"}


def p_texture(c):
    if c.cut is None or c.cut > 16000:
        return None
    fl = measure.hf_flatness(c.out, c.cut)["flatness"]
    applied = (c.D.get("codec_repair") or {}).get("applied")
    if fl >= 0.2:
        return {"id": "texture", "score": 0.0, "hypothesis": "the codec's sparse texture is left",
                "evidence": f"flatness {fl} under the cut-off (healthy 0.28-0.41): dense"}
    return {"id": "texture", "score": 0.8, "hypothesis": "the codec's sparse texture is left",
            "evidence": f"flatness {fl} under the cut-off (healthy 0.28-0.41)"
                        + ("" if applied else "; codec repair did not run (Apollo missing or off): install it"),
            "variant": None}


def p_presence(c):
    tw = measure.tonal_deviation(c.out, c.prof["ltas"], c.cut)
    bb = {int(float(k)): v for k, v in tw["by_band"].items()}
    hot = {k: v for k, v in bb.items() if 2500 <= k <= 8500}
    worst = max(hot.values()) if hot else 0.0
    return {"id": "presence", "score": float(np.clip(worst / 4, 0.0, 1.0)),
            "hypothesis": "too much 3-8 kHz (presence, sibilance)",
            "evidence": f"3-8 kHz re the reference: {', '.join(f'{k}: {v:+.1f}' for k, v in sorted(hot.items()))} dB",
            "variant": {"music.deess.max_cut_db": 12}, "label": "stronger de-esser (up to 12 dB)"}


def p_separation(c):
    moves = [abs(b.get("move_db", 0.0)) for b in (c.S.get("balance") or []) if "move_db" in b]
    big = max(moves) if moves else 0.0
    sm = c.music.get("stem_match") or {}
    if not sm.get("enabled") and big < 2:
        return None
    return {"id": "separation", "score": float(np.clip(0.2 + big / 10, 0.0, 0.8)),
            "hypothesis": "separation artefacts exposed by big per-instrument changes",
            "evidence": f"largest stem balance move {big:.1f} dB; per-stem EQ "
                        f"{'on' if sm.get('enabled') else 'off'} at {sm.get('alpha')}",
            "variant": {"music.stem_match.alpha": 0.4, "music.balance.limit_db": 1.5},
            "label": "gentler per-instrument changes"}


def p_width(c):
    so, si = measure.stereo(c.out), measure.stereo(c.orig)
    wm = c.music.get("width_match") or {}
    score = 0.2
    if so.get("mono_loss_db", 0) < -1.0:
        score += 0.3
    var, lab = None, None
    if wm.get("enabled"):
        t = dict(wm.get("targets") or {})
        var = {"music.width_match.targets": {k: v - 5 for k, v in t.items()}}
        lab = "vocal, drums and bass 5 dB narrower"
    elif (c.music.get("widen") or {}).get("enabled"):
        var, lab, score = {"music.widen.enabled": False}, "no extra width", score + 0.2
    return {"id": "width", "score": score, "hypothesis": "the stereo image: too wide or not centred",
            "evidence": f"side re mid {si.get('side_minus_mid_db')} -> {so.get('side_minus_mid_db')} dB, mono loss "
                        f"{si.get('mono_loss_db')} -> {so.get('mono_loss_db')} dB",
            "variant": var, "label": lab}


def p_limiting(c):
    lim = c.M.get("limiter") or {}
    o1, act = lim.get("over1_pct", 0.0), lim.get("active_pct", 0.0)
    ms = c.P.get("master") or {}
    score = float(np.clip(o1 / 10 + act / 200, 0.0, 0.9))
    var = ({"master.over1_max_pct": 2.5, "master.over2_max_pct": 0.5} if ms.get("over1_max_pct", 0) > 2.5 else
           {"master.over1_max_pct": 1.5, "master.over2_max_pct": 0.3})
    return {"id": "limiting", "score": score, "hypothesis": "the master limiter works audibly (pumping, flattening)",
            "evidence": f"limiter above 1 dB {o1:.1f} % of the time, active {act:.0f} %",
            "variant": var, "label": "a quieter master with an almost idle limiter"}


def p_dynamics(c):
    led = c.S.get("ledger") or []
    if len(led) < 2:
        return None
    from .review import _remedy
    drops = sorted([(led[i]["step"], led[i]["crest3s_db"] - led[i - 1]["crest3s_db"]) for i in range(1, len(led))],
                   key=lambda s: s[1])
    step, d = drops[0]
    fx = _remedy(step, c.P, c.S)
    return {"id": "dynamics", "score": float(np.clip(-d / 1.5, 0.0, 0.9)),
            "hypothesis": "a step that flattened the hits",
            "evidence": f"3 s crest {led[0]['crest3s_db']} -> {led[-1]['crest3s_db']} dB; the biggest loss: {step} "
                        f"{d:+.2f} dB",
            "variant": fx, "label": f"undo what {step} did"}


def p_lows(c, want):
    tw = measure.tonal_deviation(c.out, c.prof["ltas"], c.cut)
    bb = {int(float(k)): v for k, v in tw["by_band"].items()}
    lowmid = np.mean([v for k, v in bb.items() if 150 <= k <= 500]) if bb else 0.0
    low = np.mean([v for k, v in bb.items() if k < 150]) if bb else 0.0
    caps = [eq for eq in (c.S.get("overall_eq") or []) if any(v >= eq["hi_db"] - 0.05 for f, v in eq["db"].items()
                                                              if int(f) <= 250)]
    if want == "thin":
        return {"id": "lows", "score": float(np.clip(-low / 8, 0.0, 0.8)), "hypothesis": "too little low end",
                "evidence": f"under 150 Hz {low:+.1f} dB re the reference" + ("; the EQ is at its +6 dB cap there"
                                                                             if caps else ""),
                "variant": {"music.match.hi_db": 9.0}, "label": "allow 3 dB more low boost (lifts noise too)"}
    return {"id": "lows", "score": float(np.clip(max(lowmid, low) / 4 + (0.3 if caps else 0.0), 0.0, 0.8)),
            "hypothesis": "too much low or low-mid end",
            "evidence": f"under 150 Hz {low:+.1f} dB, 150-500 Hz {lowmid:+.1f} dB re the reference"
                        + ("; the EQ boosted to its +6 dB cap" if caps else ""),
            "variant": {"music.match.hi_db": 3.0}, "label": "low boosts capped at 3 dB"}


PROBES = {
    "digital": [p_whistles, p_top, p_texture],
    "harsh": [p_presence, p_whistles, p_top],
    "scattered": [p_top, p_separation, p_width, p_whistles],
    "processed": [p_separation, p_limiting, p_whistles, p_top],
    "muddy": [lambda c: p_lows(c, "muddy")],
    "boomy": [lambda c: p_lows(c, "boomy")],
    "thin": [lambda c: p_lows(c, "thin")],
    "squashed": [p_dynamics, p_limiting],
}


def _link_inputs(work, sub):
    os.makedirs(sub, exist_ok=True)
    skip = {"restored.wav", "restored.m4a", "AB.wav", "compare.png", "verification.md", "verification.json",
            "master.json", "steps.json", "review.md", "plan_used.json", "log.txt", "vocal_eq.json", "previous",
            "autofix_rejected", "investigate", "commands.jsonl", "run_record.zip", "apollo_chunks"}
    for name in os.listdir(work):
        if name in skip or name.startswith("."):
            continue
        dst = os.path.join(sub, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(work, name)), dst)


def run(work, complaint, render=True, max_variants=3, log=print):
    cats, vocal = categories(complaint)
    log(f"[investigate] complaint: {complaint!r} -> {', '.join(cats)}" + (" (about the vocal)" if vocal else ""))
    c = Context(work)
    found, seen = [], set()
    for cat in cats:
        for probe in PROBES[cat]:
            key = getattr(probe, "__name__", "") + cat
            if key in seen:
                continue
            seen.add(key)
            try:
                r = probe(c)
            except Exception as e:                                         # noqa: BLE001
                log(f"[investigate] probe {getattr(probe, '__name__', '?')} failed: {e}")
                continue
            if r and r["id"] not in [x["id"] for x in found]:
                found.append(r)
    found.sort(key=lambda r: -r["score"])
    for r in found:
        log(f"[investigate] {r['score']:.2f}  {r['hypothesis']}: {r['evidence']}")
    variants = []
    for r in found:
        if r["score"] >= 0.2 and r.get("variant") and r["variant"] not in [v["set"] for v in variants]:
            variants.append({"set": r["variant"], "label": r["label"], "why": r["hypothesis"]})
        if len(variants) >= max_variants:
            break
    for r in found:                                   # a bracket shows whether the cause is there at all
        if len(variants) < max_variants and r.get("bracket") and r["score"] >= 0.2 \
                and r["bracket"] not in [v["set"] for v in variants]:
            variants.append({"set": r["bracket"], "label": r["bracket_label"], "why": r["hypothesis"] + " (bracket)"})
    out_dir = os.path.join(work, "investigate")
    os.makedirs(out_dir, exist_ok=True)
    if render and variants:
        procs = []
        for i, v in enumerate(variants, 1):
            sub = os.path.join(out_dir, f"v{i}")
            _link_inputs(work, sub)
            sets = [f"{k}={json.dumps(val)}" for k, val in v["set"].items()]
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                "audiofix.py"), "restore", "--work", sub, "--plan",
                   os.path.join(work, "plan_used.json"), "--no-autofix", "--no-plots",
                   "--why", f"investigate variant {i} for '{complaint}': {v['label']} (tests: {v['why']})",
                   "--set"] + sets
            log(f"[investigate] rendering variant {i}: {v['label']} ({', '.join(sets)})")
            procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(out_dir, f"v{i}.log"), "w"),
                                          stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        for i, v in enumerate(variants, 1):
            sub = os.path.join(out_dir, f"v{i}")
            vv = _load(os.path.join(sub, "verification.json"))
            v["checks"] = (f"{sum(ch['pass'] for ch in vv.get('checks', []))}/{len(vv.get('checks', []))}"
                           if vv.get("checks") else "render failed: see " + os.path.join(out_dir, f"v{i}.log"))
            v["file"] = os.path.join(sub, "restored.wav")
        zh = any("\u3040" <= ch <= "\u9fff" for ch in complaint or "")
        _listening_file(work, c, variants, out_dir, vocal, log, zh=zh)
        with open(os.path.join(out_dir, "variants.json"), "w") as fh:
            json.dump({"complaint": complaint, "variants": variants}, fh, ensure_ascii=False, indent=1)
    _write(work, complaint, cats, found, variants, out_dir, getattr(c, "where", None))
    log(f"[next] send {os.path.join(out_dir, 'ABC.wav')} with the order in investigate.md (current render, then each "
        f"variant). Tell the listener the evidence for each hypothesis in plain words, ask which version is best, then "
        f"apply its settings with restore --work {work} --set ... --why \"the listener chose version N: ...\" "
        f"(investigate.md lists them)")
    return found, variants


def _passages(c, vocal, n=4):
    picks = list((c.V.get("ab_times_s") or [])[:n])
    tl = c.D.get("timeline") or []
    if vocal and tl and "vocals" in tl[0]:
        best = sorted(tl, key=lambda r: -r["vocals"])[:2]
        picks = [r["t0"] + 4 for r in best] + picks
    out = []
    for t in picks:
        if all(abs(t - u) > 8 for u in out) and t + 7 < len(c.out) / SR:
            out.append(float(t))
    return out[:n]


def _label(i, zh):
    """The version number, spoken (espeak-ng) before its clip; without espeak-ng, i + 1 short beeps. A listener could
    not tell four back-to-back clips apart without it."""
    try:
        import io
        import soundfile as sf
        from scipy.signal import resample_poly
        word = ["零", "一", "二", "三", "四", "五"][i] if zh else ["zero", "one", "two", "three", "four", "five"][i]
        p = subprocess.run(["espeak-ng", "-v", "cmn" if zh else "en", "-s", "160", "--stdout", word],
                           capture_output=True, check=True, timeout=20)
        z, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=True)
        z = resample_poly(z[:, 0], SR, sr).astype(np.float32)
    except Exception:                                                  # noqa: BLE001
        t = np.arange(int(0.08 * SR)) / SR
        beep = (np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t))).astype(np.float32)
        z = np.concatenate([np.concatenate([beep, np.zeros(int(0.08 * SR), np.float32)]) for _ in range(i + 1)])
    z = z / (np.abs(z).max() + 1e-9) * 0.25
    return np.concatenate([np.stack([z, z], axis=1), np.zeros((int(0.3 * SR), 2), np.float32)])


def _listening_file(work, c, variants, out_dir, vocal, log, zh=True):
    files = [c.out] + [audio_io.load(v["file"]) for v in variants if os.path.exists(v.get("file", ""))]
    picks, Lp = _passages(c, vocal), 7.0
    fade = np.minimum(1, np.minimum(np.arange(int(Lp * SR)), np.arange(int(Lp * SR))[::-1]) / (0.05 * SR))[:, None]
    parts, pos, where = [], 0, []
    for t0 in picks:
        s0, s1 = int(t0 * SR), int((t0 + Lp) * SR)
        ref = loudness.integrated(files[0][s0:s1])
        for i, y in enumerate(files):
            lab = _label(i, zh)
            seg = y[s0:s1] * float(10 ** ((ref - loudness.integrated(y[s0:s1])) / 20)) * fade
            where.append((t0, i, (pos + len(lab)) / SR))
            for piece in (lab, seg, np.zeros((int(0.6 * SR), 2), np.float32)):
                parts.append(piece)
                pos += len(piece)
        parts.append(np.zeros((int(1.4 * SR), 2), np.float32))
        pos += int(1.4 * SR)
    ab = np.vstack(parts).astype(np.float32)
    ab *= min(1.0, 0.89 / (np.abs(ab).max() + 1e-9))
    audio_io.save_wav(os.path.join(out_dir, "ABC.wav"), ab, bits=16)
    c.picks, c.where = picks, where
    log(f"[investigate] wrote {os.path.join(out_dir, 'ABC.wav')}: passages {picks}; each version is announced by its "
        f"number, in order: current, " + ", ".join(f"variant {i}" for i in range(1, len(files))))
    return where


def _write(work, complaint, cats, found, variants, out_dir, where=None):
    L = [f"# Investigation: \"{complaint}\"", "", f"Read as: {', '.join(CATEGORIES[c][0] for c in cats)}.", "",
         "## Hypotheses, strongest evidence first", ""]
    for r in found:
        L.append(f"- **{r['hypothesis']}** (evidence {r['score']:.2f}): {r['evidence']}")
    L += ["", "## Versions in ABC.wav (per passage, in this order)", "", "| # | version | settings | checks |",
          "|---|---|---|---|", "| 0 | current render | - | see verification.md |"]
    for i, v in enumerate(variants, 1):
        L.append(f"| {i} | {v['label']} (tests: {v['why']}) | `" + " ".join(f"{k}={json.dumps(val)}"
                                                                          for k, val in v['set'].items())
                 + f"` | {v.get('checks', 'not rendered')} |")
    if where:
        L += ["", "Where each version starts in ABC.wav (m:ss; each clip is 7 s and is announced by its number):", "",
              "| passage (s in the song) | " + " | ".join(f"version {i}" for i in range(len(variants) + 1)) + " |",
              "|---|" + "---|" * (len(variants) + 1)]
        for t0 in sorted({w[0] for w in where}, key=lambda t: [w[0] for w in where].index(t)):
            cells = [f"{int(s // 60)}:{int(s % 60):02d}" for t, i, s in where if t == t0]
            L.append(f"| {t0:.0f} | " + " | ".join(cells) + " |")
        L += ["", "Full-length versions: `investigate/v<number>/restored.wav` (and .m4a); version 0 is `restored.wav`."]
    L += ["", "## Next", "", "1. Send ABC.wav. Tell the listener, in plain words, which causes the measurements point "
          "to and what each version changes.", "2. Ask which version sounds best, or whether none fixes it.",
          f"3. Apply the chosen settings: `restore --work {work} --set <settings> --why \"the listener chose "
          f"version N: <their words>\"` and deliver as usual.",
          "4. If none helps, ask what exactly they hear (where, which instrument) and run investigate again with "
          "their words."]
    with open(os.path.join(out_dir, "investigate.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
