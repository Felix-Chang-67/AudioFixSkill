"""Diagnosis: measure everything the plan rules need, globally and per content class (music / speech)."""
import json, os, time
import numpy as np

from . import SR, audio_io, delimit, dsp, loudness, measure, profiles, segments, stems as S


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
    return {"ffmpeg": audio_io.have_ffmpeg(), "stems": S.available(), "delimiter": delimit.available(),
            "numba": dsp.HAVE_NUMBA, "asr": measure.asr_available()}


def cat_frames(act, regs, hop):
    parts = [act[int(a / hop):int(b / hop)] for a, b in regs if b - a > 0.5]
    return np.concatenate(parts) if parts else act[:0]


def run(input_path, work, content="auto", music_profile="jrock_band", speech_profile="speech_ltass",
        use_stems=True, log=print):
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
         "profiles": {"music": music_profile, "speech": speech_profile}}
    log(f"[analyze] {n / SR:.1f} s, {src.get('codec')} {src.get('bit_rate', 0) // 1000} kb/s; capabilities {cap}")

    D["loudness"] = loudness.summary(x)
    D["clipping"] = measure.clipping(x)
    D["dc_dbfs"] = measure.dc_offset(x)
    D["hum"] = measure.hum(x)
    D["bandwidth"] = measure.bandwidth(x)
    D["codec_notches"] = measure.codec_notches(x)
    D["noise"] = measure.noise_floor(x)
    D["stereo"] = measure.stereo(x)
    log(f"[analyze] global: {D['loudness']}, cutoff {D['bandwidth']['cutoff_hz']} Hz, stereo {D['stereo'].get('kind')}")

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

    mus = regions(proc, "music")
    spc = regions(proc, "speech")
    if mus:
        xm = cat(x, mus)
        M = {"duration_s": round(len(xm) / SR, 1), "loudness": loudness.summary(xm), "stereo": measure.stereo(xm)}
        prof = profiles.load(music_profile)
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
    D["analysis_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(work, "diagnosis.json"), "w") as fh:
        json.dump(D, fh, ensure_ascii=False, indent=1, default=float)
    return D
