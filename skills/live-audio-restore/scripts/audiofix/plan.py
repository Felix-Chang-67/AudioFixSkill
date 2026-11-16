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
         "reference": D.get("reference"),
         "notes": [], "repair": {}, "music": None, "speech": None}
    if not has_stems:
        P["notes"].append("no stem separation available: stem-level steps fall back to mix-level ones "
                          "(install demucs + torch for the full treatment)")
    if D.get("content_warning"):
        P["notes"].append(D["content_warning"])

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
    cr = D.get("codec_repair", {})
    R["codec"] = {"enabled": bool(cr.get("applied")),
                  "why": (cr.get("why", "") + (": repaired with Apollo during analyze (input_repaired.wav); "
                                               "set false to render from the original" if cr.get("applied") else ""))
                  or "not measured"}

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
            def _anti(s):
                return s.get("anti_pct", 0) >= 10 or (s.get("coherent_anti_pct", 0) >= 5
                                                      and s.get("mono_loss_db", 0) < -1.0)
            flagged = [k for k, v in st.items() if _anti(v["stereo"])]
            info = ", ".join(f"{k} {v['stereo'].get('anti_pct', 0)} % / {v['stereo'].get('coherent_anti_pct', '-')} % / "
                             f"{v['stereo'].get('mono_loss_db', '-')} dB" for k, v in st.items())
            M["antiphase"] = {"stems": flagged,
                              "why": f"per stem: energy in 4 s windows with L-R correlation < -0.3 / energy in coherent "
                                     f"anti-phase bins / mono-sum loss: {info}. Rule: repair at >= 10 % of windows "
                                     f"(a whole layer reversed), or >= 5 % coherent with mono loss beyond -1 dB (anti-phase "
                                     f"in some bands only); only the anti-phase bins are rotated"}
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
        drum_corr = (st or {}).get("drums", {}).get("stereo", {}).get("corr_median", 1.0)
        diffuse = drum_corr < 0.8
        M["widen"] = {"enabled": bool(st) and kind == "stereo" and not diffuse, "side_db": 3.0,
                      "why": f"{kind} source, drum stem's median L-R correlation {drum_corr}"
                             + (": no synthetic width" if kind != "stereo" else
                                " (< 0.8: a diffuse, room-miked image, already wide): no extra width" if diffuse else
                                ": a mixed image; the guitar/keys layer gets +3 dB of side above 150 Hz after the "
                                "polarity repair")}
        prof = profiles.load(D["profiles"]["music"])
        pst = bool(st) and bool(prof.get("stems"))
        M["stem_match"] = {"enabled": pst, "alpha": 0.6, "lo_db": -6.0, "hi_db": 3.0, "hi_low_db": 4.0,
                           "vocal_presence_db": 6.0,
                           "why": "each stem 60 % toward the same stem of the reference (local fixes: harsh vocal, "
                                  "cymbals, bleed); boosts capped (+3 dB, the vocal's 1-8 kHz presence +6 dB, always "
                                  "scaled by how much the stem owns the band), bass/kick never thinned below 120 Hz"
                           if pst else "needs stems and a profile with stem statistics"}
        # room-miked images: the stems' widths toward the reference stems' (studio vocals, bass and kick are mono)
        WIDTH_DEFAULT = {"vocals": -30.0, "drums": -22.0, "bass": -40.0}
        if st and diffuse and kind == "stereo":
            targets = {s: (prof.get("stems", {}).get(s, {}).get("side_minus_mid_db", WIDTH_DEFAULT[s]))
                       for s in WIDTH_DEFAULT}
            M["width_match"] = {"enabled": True, "targets": targets, "max_cut_db": 20.0,
                                "why": f"room-miked image (drums' median L-R correlation {drum_corr} < 0.8): the "
                                       f"vocal, drum and bass stems carry the room's width. Each is narrowed toward "
                                       f"the reference stem's side level re mid {targets} dB, at most 20 dB; the "
                                       f"guitars/keys keep their width"}
        else:
            M["width_match"] = {"enabled": False, "why": "mixed (board or broadcast) image, or no stems: widths kept"}
        if pst:
            live = {k: v["level_re_mix_db"] for k, v in st.items()}
            ref = {k: round(v["ratio_db"], 1) for k, v in prof["stems"].items()}
            frac, lim = (0.75, 6.0) if diffuse else (0.5, 3.0)
            M["balance"] = {"enabled": True, "fraction": frac, "limit_db": lim, "absent_db": -30.0,
                            "why": f"stem levels re mix, live {live} vs reference {ref}: move {frac:.0%} of the way, "
                                   f"+-{lim:.0f} dB max" + (" (room-miked: the bass is far, the room is loud)"
                                                            if diffuse else "")
                                   + ". Measured where the tone is measured: a stem below -30 dB re mix there is "
                                     "absent and left alone, and without drums there the arrangement is acoustic or "
                                     "stripped down and nothing is moved (the full band's ratios do not apply)"}
        else:
            M["balance"] = {"enabled": False, "why": "needs stems and a profile with stem statistics"}
        if limited:
            M["dynamics"] = {"enabled": bool(st), "drum_ceiling_over_lufs": 11.0,
                             "bass": [3.0, 10, 160, 70], "vocals": [3.0, 4, 90, 60], "other": [2.5, 8, 120, 60],
                             "bus": [1.8, 25, 250, 60],
                             "why": f"{dyn}: limited source. Peaks are controlled per stem (drum hits the de-limiter "
                                    f"brought back shaved at drum loudness +11 dB, bass/vocal/guitars compressed "
                                    f"separately) so the master limiter does not pump the whole mix"}
        else:
            M["dynamics"] = {"enabled": bool(st), "drum_ceiling_over_lufs": 15.0,
                             "bass": [2.0, 10, 160, 70], "vocals": [2.0, 4, 90, 60], "other": [1.5, 8, 120, 60],
                             "bus": [1.0, 25, 250, 60],
                             "why": f"{dyn}: open recording, its dynamics are part of the performance. Light per-stem "
                                    f"control only: drum hits shaved at drum loudness +15 dB (at +11 dB an open "
                                    f"recording lost >1 dB on 18 % of the time and 1.8 dB of crest), 2:1 / 1.5:1 "
                                    f"compression, no bus glue"}
        M["macro"] = {"enabled": not limited, "max_db": 6.0,
                      "why": f"{dyn}: " + ("open recording: after processing, the original's 3 s loudness contour is put "
                                           "back (slow gain, within +-6 dB), so quiet intros and breaks stay quiet; tone "
                                           "and stem changes alone took 2 LU of range off an open recording"
                                           if not limited else "limited source: its flattened contour is not kept")}
        M["match"] = {"enabled": True, "alpha": 0.85, "lo_db": -8.0, "hi_db": 6.0,
                      "why": f"long-term spectrum deviates {mu['tonal']['rms_db']} dB RMS from the profile "
                             f"(by band: {mu['tonal']['by_band']})"}
        rs = D.get("reference_section")
        if rs:
            M["match"]["measure_section"] = [rs["start_s"], rs["end_s"]]
            M["match"]["why"] += (f". The reference is this song's studio version; its passage was found at "
                                  f"{rs['start_s']}-{rs['end_s']} s (chroma similarity {rs['similarity']}). Tone and "
                                  f"stem balance are measured there, like with like, and the fixed EQ applies to the "
                                  f"whole song (the song's other parts may differ by 8 dB in the low end)")
        bw = D["bandwidth"]
        damaged = bool(D.get("codec_repair", {}).get("codec_damage"))
        ext = (bw["brick_wall"] and bw["cutoff_hz"] < 18500) or damaged
        # a codec low-pass with a transition band (no brick wall) is already down at the measured cut-off: start lower
        ext_cut = bw["cutoff_hz"] if bw["brick_wall"] else max(8000, bw["cutoff_hz"] - 800)
        M["band_ext"] = {"enabled": bool(ext), "cutoff_hz": ext_cut, "below_ref_db": 0.0, "noise_mix": 0.5,
                         "why": f"content stops at {bw['cutoff_hz']} Hz with a {bw['edge_drop_db']} dB edge"
                                + (" and a codec's sparse texture below it" if damaged else "")
                                + (" (codec low-pass): rebuild the top from the band below; half of it as a "
                                   "noise-like copy with the same envelope, because a shifted codec band is sparse "
                                   "and sounds metallic while real air is dense" if ext else
                                   ": no codec wall, nothing to rebuild")}
        M["mono_bass_hz"] = 100 if kind == "stereo" else None
        P["music"] = M

    # ---------------------------------------------------------------- speech (MC) path
    sp = D.get("speech")
    if sp:
        m = sp["metrics"]
        S = {"mode": {"value": "v2",
                      "why": "the talker's voice gets the music path's vocal treatment (v2): voice EQ toward the "
                             "reference vocal at the music path's strength, its vocal compressor, de-limiting and "
                             "band extension when measured; 'gentle' = half-way to the speech spectrum, +3 dB boosts"}}
        plr = sp.get("loudness", {}).get("plr_db", 99)
        songs_limited = bool(P.get("music")) and bool(P["music"]["delimit"]["enabled"])
        S["delimit"] = {"enabled": bool(songs_limited and plr < 11), "depth": 1.5,
                        "why": f"spoken parts: peak-to-loudness {plr} dB; songs in this file "
                               + ("de-limited" if songs_limited else "not limited or absent")
                               + ". Rule: de-limit the talk when the songs show the broadcast limiter (same chain) and "
                                 "the talk's PLR is < 11 dB. A low PLR alone is not proof: steady crowd noise gives one "
                                 "too. For a broadcast MC on its own, set speech.delimit.enabled=true"}
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
        mprof = profiles.load(D["profiles"]["music"])
        song_chain = bool(P.get("music")) and bool(P["music"].get("stem_match", {}).get("enabled"))
        target = "song_vocal" if song_chain else "studio_vocal" if mprof.get("stems", {}).get("vocals") else "ltass"
        why_t = {"song_vocal": "the EQ the music path gives the singer's vocal stem in this file (same mic, PA and "
                               "broadcast chain, so the same correction)",
                 "studio_vocal": f"the studio vocal of the '{D['profiles']['music']}' reference above 300 Hz, the "
                                 f"long-term speech spectrum below (speaking sits lower than singing)",
                 "ltass": "the long-term speech spectrum (the music profile has no vocal statistics)"}[target]
        S["match"] = {"enabled": True, "target": target, "alpha": 0.85, "lo_db": -8.0, "hi_db": 3.0, "cutoff_hz": vb,
                      "chain_file": None,
                      "why": f"voice deviates {sp['tonal']['rms_db']} dB RMS from the speech spectrum "
                             f"(by band: {sp['tonal']['by_band']}). Target: {why_t}. 85 % of the way like the music "
                             f"path's overall match; boosts only where the voice dominates the mix and at most +3 dB "
                             f"(big low boosts on a PA'd voice bring up the room), none above the voice's {vb} Hz"}
        S["vad"] = {"value": sp.get("vad", {}).get("method"), "why": "talker located with "
                    + ("Whisper word timings (crowd shouting has no words)" if sp.get("vad", {}).get("method") == "asr"
                       else "voicing/harmonicity (install faster-whisper for word-accurate timing)")}
        S["deess"] = {"enabled": True, "pct": 80, "max_cut_db": 6, "why": "tame sibilance after the presence lift"}
        S["compress"] = {"ratio": 3.0, "att_ms": 4, "rel_ms": 90, "pct": 60,
                         "why": "the music path's vocal compressor: steady, forward voice"}
        S["leveler"] = {"enabled": True, "range_db": 6.0, "window_s": 3.0,
                        "why": "even out loud and quiet sentences (slow, +-6 dB)"}
        gap = m.get("gap_minus_speech_db")
        S["crowd"] = {"enabled": gap is not None and gap > -8, "cap_db": 2.0, "floor_db": 18.0, "gap_db": 4.0,
                      "why": f"between sentences the crowd sits {gap} dB re speech (rule: > -8 dB means the crowd "
                             f"competes with the talk; cap it outside speech, never touch speech by more than 1.5 dB)"}
        bw = D["bandwidth"]
        sext = bool(bw["brick_wall"] and bw["cutoff_hz"] < 18500)
        S["band_ext"] = {"enabled": sext, "cutoff_hz": bw["cutoff_hz"], "below_ref_db": 3.0,
                         "why": f"content stops at {bw['cutoff_hz']} Hz with a {bw['edge_drop_db']} dB wall"
                                + (" (codec low-pass): rebuild the top as for the songs, 3 dB gentler (a loud synthetic "
                                   "top on a voice can lisp)" if sext else ": no codec wall, nothing to rebuild")}
        P["speech"] = S

    # ---------------------------------------------------------------- master
    ml = (D.get("music") or {}).get("loudness")
    music_open = bool(ml) and not (ml["lra_lu"] < 4 or ml["plr_db"] < 11)
    o1, o2 = (2.5, 0.5) if music_open else (8.0, 1.5)
    min_crest = round(ml["crest3s_db"] - 0.5, 2) if music_open else None
    # a quiet, peaky open recording may only keep its crest below -14 LUFS: search down to the original's loudness
    # (-0.5 dB, verify's "not quieter" limit), never below -20
    floor = min(-14.0, max(-20.0, round(ml["lufs"] - 0.5, 1))) if music_open else -14.0
    P["master"] = {"ceiling_dbtp": -1.0, "over1_max_pct": o1, "over2_max_pct": o2, "search_lufs": [-8.0, floor],
                   "min_crest_db": min_crest, "speech_only_lufs": -16.0, "speech_offset_db": 4.0,
                   "why": f"music: the loudest level at which the limiter works above 1 dB at most {o1} % and above 2 dB "
                          f"at most {o2} % of the time ("
                          + ("open recording: the limiter stays almost idle, the dynamics are kept"
                             + f", and the 3 s crest kept at {min_crest} dB or more (the original's minus 0.5 dB)"
                             if music_open else "limited source: as loud as a modern master without audible limiting")
                          + "); speech-only: -16 LUFS; spoken parts of a concert sit 4 dB under the music"}
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
