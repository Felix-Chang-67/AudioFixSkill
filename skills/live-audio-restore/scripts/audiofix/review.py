"""What a careful engineer checks before delivering, computed so that every assistant sees it.

`material(D)`: things in the recording that the rules may not fit (after analyze, before rendering): applause,
a missing rhythm section or bass, extreme peaks, a weak or generic tone reference, rebuilt highs.
`render(...)`: where each WARN came from (the step ledger in steps.json), steps that ran into their limits, the edits
to the default plan, and what changed since the previous render.

Both are written to review.md and printed as [review] lines. The delivery answers every one: what it is, what was
done, and what the listener may want. Assistants that stopped at "all checks PASS" missed all of these."""
import json, os

import numpy as np

# the knob that moves each step, for a WARN traced to it
KNOBS = {
    "de-limit": "music.delimit.depth (lower), or music.delimit.enabled=false",
    "polarity": "music.antiphase.stems",
    "width": "music.width_match.max_cut_db, or music.width_match.enabled=false",
    "de-ess": "music.deess.max_cut_db (lower), or music.deess.enabled=false",
    "drum transients": "music.transients.max_db",
    "widen": "music.widen.side_db",
    "stem EQ": "music.stem_match.alpha or hi_db, or music.stem_match.enabled=false",
    "balance": "music.balance.fraction or limit_db, or music.balance.enabled=false",
    "stem dynamics": "music.dynamics: a ratio of 1.0 turns one off, e.g. music.dynamics.vocals=[1.0,4,90,60]; "
                     "the drum peak limit is music.dynamics.drum_ceiling_over_lufs",
    "bus glue": "music.dynamics.bus",
    "overall EQ": "music.match.alpha or hi_db",
    "band extension": "music.band_ext.below_ref_db (higher = quieter top), or music.band_ext.enabled=false",
    "mono bass": "music.mono_bass_hz",
    "loudness contour": "music.macro.max_db, or music.macro.enabled=false",
    "master": "master.over1_max_pct / over2_max_pct (lower = quieter, less limiting)",
}
# verify check -> (ledger measure, direction that is bad)
WARN_MEASURE = {
    "music: dynamics not squashed further": ("crest3s_db", -1),
    "music: the song's quiet and loud parts kept apart": ("lra_lu", -1),
    "music: closer to the reference tone": ("tone_rms_db", +1),
    "music: mono compatible": ("mono_loss_db", -1),
    "music: not quieter than the original": ("lufs", -1),
}


def _stretches(rows):
    out = []
    for r in rows:
        if out and abs(out[-1][1] - r["t0"]) < 1e-6:
            out[-1][1] = r["t1"]
        else:
            out.append([r["t0"], r["t1"]])
    return out


def _music_rows(D):
    """Timeline windows that lie mostly in music: talk has no drums and applause between sentences by nature."""
    segs = D.get("processing_segments") or []
    rows = []
    for r in D.get("timeline") or []:
        ov = sum(max(0.0, min(b, r["t1"]) - max(a, r["t0"])) for a, b, c in segs if c == "music")
        if ov >= 0.5 * (r["t1"] - r["t0"]):
            rows.append(r)
    return rows


def material(D):
    items = []
    tl = _music_rows(D) if D.get("music") else []
    has_stems = bool(tl) and "vocals" in tl[0]
    clap = [r for r in tl if has_stems and r["vocals"] <= -15 and r["drums"] >= -6 and r["peak_over_rms_db"] >= 20]
    for a, b in _stretches(clap):
        pk = max(r["peak_over_rms_db"] for r in clap if a <= r["t0"] < b)
        items.append({"kind": "applause", "text": (
            f"{a:.0f}-{b:.0f} s: no vocal, the drum stem carries it, and its peaks stand {pk:.0f} dB over its level: "
            f"probably applause or clapping, not the song. It is processed with the music (the drum stem's peak limit "
            f"holds the claps back). Tell the listener; they may want it kept, softened or cut")})
    song = [r for r in tl if r not in clap and r["lufs"] > -60]
    if has_stems and song:
        dr = float(np.median([r["drums"] for r in song]))
        ba = float(np.median([r["bass"] for r in song]))
        if dr < -30:
            items.append({"kind": "arrangement", "text": (
                f"No drums in the performance (median {dr:.0f} dB re mix outside applause): an acoustic or stripped-"
                f"down arrangement. The studio reference is probably a full band: stem balance does nothing here, and "
                f"its low end (bass and kick) cannot be reached by EQ. Say so, and read the overall EQ's low boosts")})
        elif ba < -30:
            items.append({"kind": "arrangement", "text": (
                f"Separation found no bass (median {ba:.0f} dB re mix): the low end can only come from EQ on the whole "
                f"mix, which lifts room and stage noise along with it. If the listener finds it thin or boomy, that is "
                f"why")})
    lu = (D.get("music") or {}).get("loudness") or {}
    if lu.get("plr_db", 0) > 18:
        pk = D.get("peaks") or []
        desc = "; ".join(f"{p['t_s']} s ({p['width_ms']:.0f} ms, {p['share_above_1khz']:.0%} above 1 kHz)" for p in pk[:4])
        items.append({"kind": "peaks", "text": (
            f"Peaks stand {lu['plr_db']} dB over the loudness (music usually 10-14 dB). Largest: {desc}. Narrow "
            f"(under 10 ms), bright peaks are claps or knocks near the recorder, and holding them back is right; drum "
            f"hits are wider. The master's loudness and the crest check depend on them")})
    ref = D.get("reference") or {}
    rs = D.get("reference_section")
    if rs and rs.get("similarity", 1) < 0.55:
        items.append({"kind": "reference", "text": (
            f"The studio version matched the recording only weakly (chroma correlation {rs['similarity']} at "
            f"{rs['start_s']}-{rs['end_s']} s; same songs scored 0.46-0.67): the live arrangement may differ from the "
            f"studio one. Its tone is still the target; check the low-end EQ and the balance against what is played")})
    if D.get("music") and ref.get("used") in ("jrock_band",) and ref.get("auto"):
        items.append({"kind": "reference", "text": f"Tone target: the generic profile. {ref.get('why', '')}"})
    cr = D.get("codec_repair") or {}
    if cr.get("applied"):
        items.append({"kind": "highs", "text": (
            f"Low-bitrate codec damage (cut-off {D['bandwidth']['cutoff_hz']} Hz, sparse highs) was repaired with "
            f"Apollo, and the band above the cut-off is rebuilt: the top octave is synthesized. Say so; if the "
            f"listener hears it as fake or harsh, music.band_ext.below_ref_db=6 makes it 6 dB quieter")})
    return items


def _flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        if k == "why":
            continue
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def plan_diff(a, b):
    """Settings that differ between two plans (repair, music, speech, master), without the explanations."""
    fa = {k: v for s in ("repair", "music", "speech", "master") for k, v in _flat(a.get(s), f"{s}.").items()}
    fb = {k: v for s in ("repair", "music", "speech", "master") for k, v in _flat(b.get(s), f"{s}.").items()}
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def render(D, P, V, steps=None, default_plan=None, prev=None):
    items = []
    led = (steps or {}).get("ledger") or []
    for c in V.get("checks", []):
        if c["pass"]:
            continue
        m = WARN_MEASURE.get(c["check"])
        if not m or len(led) < 2:
            items.append({"kind": "warn", "text": f"WARN '{c['check']}': {c.get('detail', '')}. No step ledger for "
                                                  f"it: see SKILL.md step 6"})
            continue
        key, bad = m
        moves = [(led[i]["step"], led[i][key] - led[i - 1][key]) for i in range(1, len(led))]
        worst = sorted([s for s in moves if s[1] * bad > 0.05], key=lambda s: -abs(s[1]))[:3]
        if worst:
            lead = worst[0][0]
            items.append({"kind": "warn", "text": (
                f"WARN '{c['check']}': {key} went {led[0][key]} -> {led[-1][key]} through the chain. The steps that "
                f"moved it: " + ", ".join(f"{s} {d:+.2f}" for s, d in worst) + f". Knob for {lead}: {KNOBS.get(lead, '?')}. "
                f"Change that one, re-render, and compare")})
        else:
            items.append({"kind": "warn", "text": (
                f"WARN '{c['check']}': no single step moved {key} much (start {led[0][key]}, end {led[-1][key]}); "
                f"the input already differs from what the check compares. See SKILL.md step 6")})
    for eq in (steps or {}).get("overall_eq") or []:
        top = [f for f, v in eq["db"].items() if v >= eq["hi_db"] - 0.05]
        if top:
            items.append({"kind": "limits", "text": (
                f"The overall EQ hit its +{eq['hi_db']:.0f} dB cap at {', '.join(top)} Hz "
                f"({eq['section'][0]:.0f}-{eq['section'][1]:.0f} s): the recording lacks more there than a fixed EQ "
                f"should add. If the performance has nothing there (no bass, no air), a bigger boost would only lift "
                f"noise. Tell the listener what stayed short")})
    for bm in (steps or {}).get("balance") or []:
        if bm.get("at_limit"):
            items.append({"kind": "limits", "text": (
                f"Stem balance moved {bm['stem']} by {bm['move_db']:+.1f} dB, its limit ({bm['live_db']} dB re mix "
                f"in {bm['section'][0]:.0f}-{bm['section'][1]:.0f} s): far from its studio share. Check it is the "
                f"instrument and not bleed or crowd")})
    if default_plan is not None:
        d = plan_diff(default_plan, P)
        if d:
            items.append({"kind": "edits", "text": "Edits to the default plan (each needs its reason in the delivery): "
                          + "; ".join(f"{k}: {a} -> {b}" for k, a, b in ((k, *v) for k, v in d.items()))})
    if prev:
        pp, pv = prev.get("plan"), prev.get("verification")
        if pp is not None:
            d = plan_diff(pp, P)
            ch = []
            if pv:
                old = {c["check"]: c for c in pv.get("checks", [])}
                for c in V.get("checks", []):
                    o = old.get(c["check"])
                    if o and o["pass"] != c["pass"]:
                        ch.append(f"{c['check']}: {'PASS' if o['pass'] else 'WARN'} -> {'PASS' if c['pass'] else 'WARN'}")
            items.append({"kind": "previous", "text": (
                "Since the previous render (kept in previous/): "
                + ("; ".join(f"{k}: {a} -> {b}" for k, (a, b) in d.items()) or "same plan")
                + (". Checks: " + "; ".join(ch) if ch else "")
                + ". Compare previous/restored.wav with restored.wav before choosing")})
    return items


def write(path, items, title):
    with open(path, "w") as fh:
        fh.write(f"# {title}\n\n")
        if not items:
            fh.write("Nothing to review.\n")
        for i, it in enumerate(items, 1):
            fh.write(f"{i}. **{it['kind']}**: {it['text']}\n")


def load_json(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None
