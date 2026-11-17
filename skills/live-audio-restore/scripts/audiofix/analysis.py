"""Diagnosis: measure everything the plan rules need, globally and per content class (music / speech)."""
import json, os, shutil, time
import numpy as np

from . import SR, audio_io, delimit, dsp, loudness, measure, profiles, reference, segments, stems as S


def regions(segs, label):
    return [(a, b) for a, b, c in segs if c == label]


def cat(y, regs):
    parts = [y[int(a * SR):int(b * SR)] for a, b in regs if b - a > 0.5]
    return np.concatenate(parts) if parts else y[:0]


def mask_of(n, regs):
    m = np.zeros(n, bool)
    for a, b in regs:
        m[int(a * SR):int(b * SR)] = True
    return m


def capabilities():
    from . import codec
    return {"ffmpeg": audio_io.have_ffmpeg(), "stems": S.available(), "delimiter": delimit.available(),
            "numba": dsp.HAVE_NUMBA, "asr": measure.asr_available(), "codec_repair": codec.available()}


def cat_frames(act, regs, hop):
    parts = [act[int(a / hop):int(b / hop)] for a, b in regs if b - a > 0.5]
    return np.concatenate(parts) if parts else act[:0]


def run(input_path, work, content="auto", music_profile=None, speech_profile="speech_ltass",
        use_stems=True, log=print, codec_repair="auto", lookup=None, degraded=None, generic_ok=None):
    """music_profile None: the studio version of this song, looked up online from the tags or lookup's
    artist / title (see reference.py), else the bundled default. lookup: {artist, title, country, lookup}."""
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    cap = capabilities()
    src = audio_io.probe(input_path)
    x = audio_io.load(input_path)
    wav = os.path.join(work, "input.wav")
    audio_io.save_wav(wav, x, bits=32)
    n = len(x)
    D = {"input": os.path.abspath(input_path), "work": os.path.abspath(work), "source": src,
         "duration_s": round(n / SR, 2), "capabilities": cap, "content_hint": content,
         "profiles": {"music": music_profile or reference.DEFAULT, "speech": speech_profile}}
    if degraded:
        D["degraded"] = degraded
    log(f"[analyze] {n / SR:.1f} s, {src.get('codec')} {src.get('bit_rate', 0) // 1000} kb/s; capabilities {cap}")
    if music_profile:
        D["reference"] = {"auto": False, "used": music_profile, "why": "chosen with --music-profile"}
    elif content == "speech":
        D["reference"] = {"auto": False, "used": reference.DEFAULT, "why": "talk only: no song to look up"}
    else:
        # the studio version of this very song is the best tonal target: look it up whenever nobody chose one
        lk = dict(lookup or {})
        D["profiles"]["music"], D["reference"] = reference.find(
            src.get("tags", {}), work, artist=lk.get("artist"), title=lk.get("title"), country=lk.get("country"),
            lookup=lk.get("lookup", True), with_stems=use_stems and cap["stems"], log=log)
    if content == "music" and D["reference"]["used"] == reference.DEFAULT and not generic_ok:
        raise reference.NoStudioReference(D["reference"].get("why", ""))      # stop before the long steps

    D["loudness"] = loudness.summary(x)
    D["clipping"] = measure.clipping(x)
    D["dc_dbfs"] = measure.dc_offset(x)
    D["hum"] = measure.hum(x)
    D["bandwidth"] = measure.bandwidth(x)
    # a codec's cut-off moves over time (VBR); the top must be rebuilt from the lowest edge, not the whole-file one
    cs = [measure.bandwidth(x[i:i + 15 * SR])["cutoff_hz"] for i in range(0, max(1, len(x) - 10 * SR), 15 * SR)
          if float(np.sqrt(np.mean(x[i:i + 15 * SR].astype(np.float64) ** 2))) > 1e-4]
    if cs:
        D["bandwidth"]["cutoff_p10_hz"] = int(np.percentile(cs, 10))
        D["bandwidth"]["cutoff_range_hz"] = [int(min(cs)), int(max(cs))]
    D["codec_notches"] = measure.codec_notches(x)
    D["noise"] = measure.noise_floor(x)
    D["stereo"] = measure.stereo(x)
    log(f"[analyze] global: {D['loudness']}, cutoff {D['bandwidth']['cutoff_hz']} Hz, stereo {D['stereo'].get('kind')}")

    # codec repair (Apollo): a low-pass below 16 kHz means a low-bitrate codec, whose sparse, peaky top would sound
    # fake once boosted or shifted. Everything after this (stems, measurements, rendering) uses the repaired signal;
    # the cut-off and the global numbers above stay those of the original, and verify compares with the original.
    bw = D["bandwidth"]
    D["hf_texture"] = measure.hf_flatness(x, bw["cutoff_hz"])
    fl = D["hf_texture"]["flatness"]
    damaged = bool(bw["cutoff_hz"] < 16000 and fl < 0.15)
    want = codec_repair == "on" or (codec_repair == "auto" and damaged)
    D["codec_repair"] = {"applied": False, "requested": codec_repair, "codec_damage": damaged,
                         "why": (f"cut-off {bw['cutoff_hz']} Hz ({bw['edge_drop_db']} dB edge); the band below it has a "
                                 f"spectral flatness of {fl} at {D['hf_texture']['band_hz']} Hz (rule: cut-off below "
                                 f"16 kHz and flatness below 0.15 = a low-bitrate codec's sparse texture; healthy bands "
                                 f"measure 0.28-0.41)" + (": codec damage" if damaged else ": no codec damage"))}
    if want and cap["codec_repair"]:
        from . import codec
        cached = os.path.join(work, "input_repaired.wav")
        if os.path.exists(cached):
            xr = audio_io.load(cached)[:n]
        else:
            from . import device
            dev = device.pick(log)
            log(f"[analyze] codec repair with Apollo on {device.describe()}"
                + (f" (about {6 * n / SR / 60:.0f} min on the CPU)" if dev == "cpu" else "") + " ...")
            t1 = time.time()
            chunks = os.path.join(work, "apollo_chunks")       # a run cut off part way resumes from here
            xr = codec.repair(x, log=log, checkpoint=chunks)
            audio_io.save_wav(cached + ".tmp.wav", xr, bits=32)
            os.replace(cached + ".tmp.wav", cached)
            shutil.rmtree(chunks, ignore_errors=True)
            D["codec_repair"]["seconds"] = round(time.time() - t1)
        D["codec_repair"].update(applied=True, model="apollo")
        x, wav = xr, cached
    elif want:
        D["codec_repair"]["why"] += "; Apollo is not installed (bash scripts/setup.sh), the band is used as it is"

    # steady narrow tones in the highs (codec artefacts, and the codec repair's own): notched out before anything is
    # boosted or copied up, or the band extension multiplies them
    D["tonal_lines"] = measure.tonal_lines(x)
    for w in measure.whistle_lines(x):                     # on-and-off codec sinusoids, by accumulated time
        if all(abs(w["hz"] - l["hz"]) > 12 for l in D["tonal_lines"]):
            D["tonal_lines"].append({"hz": w["hz"], "excess_db": w["excess_db"], "presence": None,
                                     "total_s": w["total_s"]})
    D["tonal_lines"].sort(key=lambda l: l["hz"])
    if D["tonal_lines"]:
        log(f"[analyze] narrow tones (codec artefacts, steady or on and off): "
            f"{[(l['hz'], l['excess_db']) for l in D['tonal_lines']]}")
    # the top band rebuilt from a shifted copy of the low band (HE-AAC band replication): every held note leaves an
    # out-of-tune copy up there
    D["sbr_copies"] = measure.sbr_copies(x, D["bandwidth"]["cutoff_hz"])
    for s in D["sbr_copies"]:
        log(f"[analyze] the top band {s['band_hz'][0]}-{s['band_hz'][1]} Hz is the low band shifted up {s['shift_hz']} Hz "
            f"(correlation {s['corr']}): copied tones (band replication)")

    st = None
    if use_stems and cap["stems"]:
        log("[analyze] separating stems (demucs htdemucs) ...")
        folder = S.separate(wav, work)
        D["stems_dir"] = folder
        st = S.load(folder, n)
        res = x - sum(st.values())
        D["separation_residual_db"] = round(float(dsp.db(np.std(res) / (np.std(x) + 1e-12))), 1)
    else:
        D["stems_dir"] = None
        log("[analyze] no stems: the plan will use the mix-only fallbacks")

    segs, feats = segments.classify(x, st, content=content)
    if st is None and content == "auto":
        # songs and talk are told apart by the rhythm section (drum + bass stems); Whisper alone cannot, lyrics are words
        D["content_warning"] = ("no stems, so songs and talk cannot be told apart and the whole file is treated as "
                                "music. For an MC / talk file rerun analyze with --content speech; for a concert with "
                                "songs and MCs, install demucs + torch")
        log(f"[analyze] WARNING: {D['content_warning']}")
    if st is not None and content == "auto" and cap["asr"]:
        # where there is no rhythm section, ask Whisper whether someone is talking (robust to separation failures)
        low = feats["rhythm"] < 0.08
        regs, i = [], 0
        while i < len(low):
            if low[i]:
                j = i
                while j < len(low) and low[j]:
                    j += 1
                if j - i >= 5:
                    regs.append((float(i), float(min(j + 2, n / SR))))
                i = j
            else:
                i += 1
        if regs:
            log(f"[analyze] checking {len(regs)} stretch(es) without a rhythm section for talk (Whisper) ...")
            _, info = measure.asr_activity(x, regs)
            wd = np.zeros(len(low))
            for t in info.get("word_times", []):
                if int(t) < len(wd):
                    wd[int(t)] += 1
            segs, feats = segments.classify(x, st, content=content, word_density=wd)
    proc = segments.assign_gaps(segs, feats.get("default", "music") if feats else "music")
    secs = segments.sections(proc, x)
    D["segments"] = [[round(a, 2), round(b, 2), c] for a, b, c in segs]
    D["processing_segments"] = [[round(a, 2), round(b, 2), c] for a, b, c in proc]
    D["sections"] = [[round(a, 2), round(b, 2)] for a, b in secs]
    log(f"[analyze] content: {D['processing_segments']}; music sections {D['sections']}")
    D["timeline"] = measure.timeline(x, st)
    D["peaks"] = measure.top_peaks(x)

    mus = regions(proc, "music")
    spc = regions(proc, "speech")
    if mus:
        xm = cat(x, mus)
        M = {"duration_s": round(len(xm) / SR, 1), "loudness": loudness.summary(xm), "stereo": measure.stereo(xm)}
        prof = profiles.load(D["profiles"]["music"])
        if prof.get("song_chroma"):
            # the reference is this very song: find its passage, so tone is compared like with like
            loc = profiles.locate(prof["song_chroma"], x, regions=mus)
            D["reference_section"] = ({"start_s": loc[0], "end_s": loc[1], "similarity": loc[2]} if loc else None)
            log(f"[analyze] reference passage in the recording: {D['reference_section'] or 'not found (no passage with chroma correlation >= 0.35 that stands out in its key)'}")
            if loc is None and D["reference"].get("auto"):
                # a looked-up preview that matches no passage: another song of the same name, or a different
                # arrangement. A wrong target is worse than the generic one
                D["reference"].update(used=reference.DEFAULT, rejected=D["profiles"]["music"],
                                      why=D["reference"]["why"] + "; but its passage was not found in the recording "
                                          "(chroma correlation < 0.35, or no clear key: another song or arrangement), "
                                          "so the bundled default profile is used")
                D["profiles"]["music"] = reference.DEFAULT
                prof = profiles.load(reference.DEFAULT)
                log(f"[reference] {D['reference']['why']}")
        M["tonal"] = measure.tonal_deviation(xm, prof["ltas"], D["bandwidth"]["cutoff_hz"])
        if st is not None:
            sm = {k: cat(v, mus) for k, v in st.items()}
            mixm = sum(sm.values()) + cat(x - sum(st.values()), mus)
            M["stems"] = {k: {"level_re_mix_db": measure.level_re_mix(v, mixm),
                              "stereo": {**measure.stem_stereo(v), **measure.coherent_antiphase(v)}}
                          for k, v in sm.items()}
            M["treble_share_pct"] = measure.treble_share(sm)
            per_sec = []
            for a, b in secs:
                seg = {k: v[int(a * SR):int(b * SR)] for k, v in st.items()}
                mx = x[int(a * SR):int(b * SR)]
                per_sec.append({"section": [a, b], **{k: measure.level_re_mix(v, mx) for k, v in seg.items()}})
            M["stem_levels_by_section"] = per_sec
        D["music"] = M
        log(f"[analyze] music: {M['loudness']}, tonal rms dev {M['tonal']['rms_db']} dB")
    if spc:
        xs = cat(x, spc)
        Sp = {"duration_s": round(len(xs) / SR, 1), "loudness": loudness.summary(xs)}
        voice_full = st["vocals"] if st is not None else x
        if st is not None:
            vs = float(np.mean(cat(st["vocals"], spc).astype(np.float64) ** 2) / (np.mean(xs.astype(np.float64) ** 2) + 1e-30))
            Sp["vocal_share"] = round(vs, 3)
            if vs < 0.3:                       # separation missed the voice: listen to the mix instead
                log(f"[analyze] vocal stem holds only {vs:.0%} of the spoken parts: locating the talker on the mix")
                voice_full = x
        log(f"[analyze] locating the talker ({'Whisper word timings' if cap['asr'] else 'harmonicity'}) ...")
        act, vad = measure.speech_activity(voice_full, spc)
        hop = 0.01
        np.save(os.path.join(work, "speech_mask.npy"), act)
        Sp["vad"] = vad
        voice = cat(voice_full, spc)
        sep_ok = st is not None and voice_full is not x
        Sp["metrics"] = measure.speech_metrics(xs, voice if sep_ok else None, cat_frames(act, spc, hop), hop)
        sprof = profiles.load(speech_profile)
        vb = Sp["metrics"].get("voice_bandwidth", {}).get("cutoff_hz")
        Sp["tonal"] = measure.tonal_deviation(voice, sprof["ltas"], vb, lo=100)
        D["speech"] = Sp
        log(f"[analyze] speech: {Sp['loudness']}, {Sp['metrics']}, talker detection {vad}")
    if generic_ok and D.get("music") and D["reference"].get("used") == reference.DEFAULT:
        D["reference"]["generic_ok"] = generic_ok
    D["analysis_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(work, "diagnosis.json"), "w") as fh:
        json.dump(D, fh, ensure_ascii=False, indent=1, default=float)
    return D
