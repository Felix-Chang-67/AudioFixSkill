"""Decision rules: diagnosis -> restoration plan.

Every switch and number in the plan is set by a measurement and carries a `why` that names it and the threshold,
so a person or another model can read, question and edit the plan before anything is rendered.
The thresholds are documented in references/diagnosis.md.
"""
import json, os

import numpy as np

from . import profiles


def _avg(d, keys):
    v = [d[k] for k in keys if k in d]
    return float(np.mean(v)) if v else 0.0


def build(D):
    cap = D["capabilities"]
    has_stems = D.get("stems_dir") is not None
    P = {"version": 1, "input": D["input"], "work": D["work"], "capabilities": cap,
         "segments": D["processing_segments"], "sections": D["sections"], "profiles": D["profiles"],
         "notes": [], "repair": {}, "music": None, "speech": None}
    if not has_stems:
        P["notes"].append("no stem separation available: stem-level steps fall back to mix-level ones "
                          "(install demucs + torch for the full treatment)")

    # ---------------------------------------------------------------- repairs on the raw mix
    R = P["repair"]
    R["highpass_hz"] = {"value": 20, "why": "remove DC and sub-sonic energy (inaudible, frees headroom)"}
    h = D["hum"]
    if h["present"]:
        harm = [k + 1 for k, e in enumerate(h["excess_db"]) if e >= 6] or [1]
        R["dehum"] = {"enabled": True, "mains_hz": h["mains_hz"], "harmonics": harm,
                      "why": f"mains peaks at {h['mains_hz']} Hz x{harm} stand {max(h['excess_db']):.0f} dB above "
                             f"the spectrum (rule: fundamental >= 10 dB or two harmonics >= 8 dB)"}
    else:
        R["dehum"] = {"enabled": False, "why": f"no mains hum (largest excess {max(h['excess_db']):.0f} dB)"}
    c = D["clipping"]
    clipped = c["flat_runs_per_min"] >= 20
    R["declip"] = {"enabled": bool(clipped and cap["ffmpeg"]),
                   "why": f"{c['flat_runs_per_min']} flat-topped runs per minute at {c['peak_dbfs']} dBFS "
                          f"(rule: >= 20/min means digital clipping)" + ("" if cap["ffmpeg"] or not clipped
                                                                         else "; ffmpeg missing, cannot declip")}

    # ---------------------------------------------------------------- music path
    mu = D.get("music")
    if mu:
        L, st = mu["loudness"], mu.get("stems")
        M = {}
        limited = L["lra_lu"] < 4 or L["plr_db"] < 11
        heavy = L["lra_lu"] < 2 or L["plr_db"] < 9.5
        dyn = f"LRA {L['lra_lu']} LU, peak-to-loudness {L['plr_db']} dB"
        if limited and cap["delimiter"]:
            M["delimit"] = {"enabled": True, "depth": 2.0 if heavy else 1.5,
                            "why": f"{dyn}: {'heavily ' if heavy else ''}limited (rule: LRA < 4 or PLR < 11; "
                                   f"depth 2 if LRA < 2 or PLR < 9.5)"}
        else:
            M["delimit"] = {"enabled": False,
                            "why": f"{dyn}: " + ("limited, but the De-limiter model is not installed: stronger "
                                                 "drum transients instead" if limited else "dynamics already open")}
        if st:
            amt, mx = ((0.35, 2.5) if M["delimit"]["enabled"] else (0.55, 4.5) if limited else (0.25, 2.0))
            M["transients"] = {"enabled": True, "amount": amt, "max_db": mx,
                               "why": "drum hits flattened by the limiter" if limited else "light lift of drum hits"}
            flagged = [k for k, v in st.items() if v["stereo"].get("anti_pct", 0) >= 10]
            info = ", ".join(f"{k} {v['stereo'].get('anti_pct', 0)} %" for k, v in st.items())
            M["antiphase"] = {"stems": flagged,
                              "why": f"share of each stem's energy in 4 s windows with L-R correlation < -0.3: {info} "
                                     f"(rule: repair a stem at >= 10 %)"}
            ts = mu.get("treble_share_pct", {}).get("vocals", 0)
            dev = _avg(mu["tonal"]["by_band"], (5040, 6350, 8000))
            strong = ts >= 50 or dev > 4
            M["deess"] = {"enabled": True, "pct": 80, "max_cut_db": 9 if strong else 6,
                          "why": f"vocal carries {ts} % of the 3-10 kHz energy, 5-8 kHz sits {dev:+.1f} dB vs profile "
                                 f"(strong if >= 50 % or > +4 dB)"}
        else:
            neg = D["stereo"].get("mid_corr_neg_pct", 0)
            M["antiphase"] = {"stems": ["mix"] if neg >= 10 else [],
                              "why": f"{neg} % of 4 s windows have negative mid-band L-R correlation (rule: >= 10 %); "
                                     f"without stems the coherent anti-phase bins of the whole mix are rotated"}
            M["transients"] = {"enabled": False, "why": "needs a drum stem"}
            M["deess"] = {"enabled": False, "why": "needs a vocal stem"}
        kind = mu["stereo"]["kind"]
        M["widen"] = {"enabled": bool(st) and kind == "stereo", "side_db": 3.0,
                      "why": f"{kind} source" + ("; the guitar/keys layer gets +3 dB of side above 150 Hz after the "
                                                "polarity repair" if kind == "stereo" else ": no synthetic width")}
        prof = profiles.load(D["profiles"]["music"])
        pst = bool(st) and bool(prof.get("stems"))
        M["stem_match"] = {"enabled": pst, "alpha": 0.6, "lo_db": -6.0, "hi_db": 3.0, "hi_low_db": 4.0,
                           "why": "each stem 60 % toward the same stem of the reference (local fixes: harsh vocal, "
                                  "cymbals, bleed); boosts capped, bass/kick never thinned below 120 Hz"
                           if pst else "needs stems and a profile with stem statistics"}
        if pst:
            live = {k: v["level_re_mix_db"] for k, v in st.items()}
            ref = {k: round(v["ratio_db"], 1) for k, v in prof["stems"].items()}
            M["balance"] = {"enabled": True, "fraction": 0.5, "limit_db": 3.0,
                            "why": f"stem levels re mix, live {live} vs reference {ref}: move halfway, +-3 dB max"}
        else:
            M["balance"] = {"enabled": False, "why": "needs stems and a profile with stem statistics"}
        M["dynamics"] = {"enabled": bool(st), "drum_ceiling_over_lufs": 11.0,
                         "bass": [3.0, 10, 160, 70], "vocals": [3.0, 4, 90, 60], "other": [2.5, 8, 120, 60],
                         "bus": [1.8, 25, 250, 60],
                         "why": "peaks are controlled per stem (drum hits shaved on the drum stem, bass/vocal/guitars "
                                "compressed separately) so the master limiter does not pump the whole mix"}
        M["match"] = {"enabled": True, "alpha": 0.85, "lo_db": -8.0, "hi_db": 6.0,
                      "why": f"long-term spectrum deviates {mu['tonal']['rms_db']} dB RMS from the profile "
                             f"(by band: {mu['tonal']['by_band']})"}
        bw = D["bandwidth"]
        ext = bw["brick_wall"] and bw["cutoff_hz"] < 18500
        M["band_ext"] = {"enabled": bool(ext), "cutoff_hz": bw["cutoff_hz"], "below_ref_db": 0.0,
                         "why": f"content stops at {bw['cutoff_hz']} Hz with a {bw['edge_drop_db']} dB wall"
                                + (" (codec low-pass): rebuild the top from the band below" if ext else
                                   ": no codec wall, nothing to rebuild")}
        M["mono_bass_hz"] = 100 if kind == "stereo" else None
        P["music"] = M

    # ---------------------------------------------------------------- speech (MC) path
    sp = D.get("speech")
    if sp:
        m = sp["metrics"]
        S = {}
        stb = m.get("speech_to_background_db")
        vsh = sp.get("vocal_share")
        use_stem = has_stems and (vsh is None or vsh >= 0.3)
        S["source"] = {"value": "stems" if use_stem else "mix",
                       "ambience_db": -12.0 if (stb is not None and stb < 6) else -6.0 if (stb or 0) >= 20 else -10.0,
                       "why": (f"voice-to-background {stb} dB: voice from the vocal stem, the rest (room, crowd) kept "
                               f"underneath at the ambience level") if use_stem else
                              (f"the vocal stem holds only {vsh:.0%} of the spoken parts (separation missed the voice): "
                               f"work on the mix" if has_stems else "no stems: work on the mix")}
        nr = 10 if (stb is not None and stb < 6) else 8 if (stb is None or stb < 15) else 6 if stb < 25 else 0
        S["denoise"] = {"enabled": nr > 0 and cap["ffmpeg"], "nr_db": nr,
                        "why": f"voice-to-background {stb} dB -> stationary noise reduction {nr} dB (ffmpeg afftdn). "
                               f"Neural denoisers are not used: on PA'd, reverberant live speech they remove speech"}
        rum = m.get("rumble_db", -40)
        S["highpass_hz"] = {"value": 100 if rum > -12 else 75,
                            "why": f"energy below 80 Hz is {rum} dB re 200-800 Hz (rule: > -12 dB means rumble)"}
        vb = m.get("voice_bandwidth", {}).get("cutoff_hz", 20000)
        S["match"] = {"enabled": True, "alpha": 0.5, "lo_db": -6.0, "hi_db": 3.0, "cutoff_hz": vb,
                      "why": f"voice deviates {sp['tonal']['rms_db']} dB RMS from the speech spectrum "
                             f"(by band: {sp['tonal']['by_band']}); half-way, boosts capped at +3 dB (a PA'd voice is "
                             f"naturally thinner, big low boosts bring up room boom), none above the voice's {vb} Hz"}
        S["vad"] = {"value": sp.get("vad", {}).get("method"), "why": "talker located with "
                    + ("Whisper word timings (crowd shouting has no words)" if sp.get("vad", {}).get("method") == "asr"
                       else "voicing/harmonicity (install faster-whisper for word-accurate timing)")}
        S["deess"] = {"enabled": True, "pct": 80, "max_cut_db": 6, "why": "tame sibilance after the presence lift"}
        S["compress"] = {"ratio": 2.5, "att_ms": 5, "rel_ms": 120, "pct": 60, "why": "steady speech level"}
        S["leveler"] = {"enabled": True, "range_db": 6.0, "window_s": 3.0,
                        "why": "even out loud and quiet sentences (slow, +-6 dB)"}
        gap = m.get("gap_minus_speech_db")
        S["crowd"] = {"enabled": gap is not None and gap > -8, "cap_db": 2.0, "floor_db": 18.0, "gap_db": 4.0,
                      "why": f"between sentences the crowd sits {gap} dB re speech (rule: > -8 dB means the crowd "
                             f"competes with the talk; cap it outside speech, never touch speech by more than 1.5 dB)"}
        S["band_ext"] = {"enabled": False, "why": f"voice bandwidth {vb} Hz: a synthetic top on a voice tends to "
                                                  f"lisp; off unless the listener asks for more air"}
        P["speech"] = S

    # ---------------------------------------------------------------- master
    P["master"] = {"ceiling_dbtp": -1.0, "over1_max_pct": 8.0, "over2_max_pct": 1.5, "search_lufs": [-8.0, -14.0],
                   "speech_only_lufs": -16.0, "speech_offset_db": 4.0,
                   "why": "music: the loudest level at which the limiter works above 1 dB at most 8 % and above 2 dB "
                          "at most 1.5 % of the time; speech-only: -16 LUFS; spoken parts of a concert sit 4 dB "
                          "under the music"}
    return P


def save(P, path):
    with open(path, "w") as fh:
        json.dump(P, fh, ensure_ascii=False, indent=1, default=float)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def override(P, assignments):
    """Apply key=value edits like music.delimit.depth=1.5 or speech.crowd.enabled=false."""
    for a in assignments or []:
        k, v = a.split("=", 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        node = P
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = val
    return P
