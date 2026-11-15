"""Speech (MC) path.

mode "v2" (default) treats the talker's voice the way the music path's strong remaster (v2) treats the vocal stem:
de-limit when the talk was limited, voice EQ toward the vocal of the reference at the music path's strength (boosts
only where the voice dominates the mix, capped), the music path's vocal compressor, band extension above a codec
wall. Speech-specific steps stay: room kept underneath, gentle stationary denoise, slow leveler, crowd taming between
sentences (never more than 1.5 dB on speech).

The voice target, in order of preference:
  song_vocal    the file also has songs: the EQ the music path gave the vocal stem is applied to the talk (same mic,
                same PA, same broadcast chain, so the same correction; the talker sounds like the restored singer)
  studio_vocal  the reference's studio vocal above 300 Hz, the long-term speech spectrum below (speech sits lower
                than singing: a studio vocal is nearly empty below 200 Hz)
  ltass         the long-term speech spectrum (no vocal statistics in the music profile)
mode "gentle" is the earlier path: half-way match to the speech spectrum, boosts at most +3 dB, 2.5:1 compression.

No neural denoisers: on PA'd, reverberant live speech they remove speech along with the noise."""
import json, os, subprocess
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import SR, audio_io, delimit, dsp, measure, profiles  # noqa: F401
from .dsp import CENT, undb

GENTLE = {"alpha": 0.5, "lo_db": -6.0, "hi_db": 3.0, "compress": (2.5, 5, 120, 60)}
VOICE_CORE = (CENT >= 300) & (CENT <= 1000)


def mode_of(S):
    m = S.get("mode", "v2")
    return m.get("value", "v2") if isinstance(m, dict) else m


def voice_target(music_prof, speech_prof):
    """Target long-term spectrum of the voice on the 1/6-octave grid, normalised at 300-1000 Hz: the studio vocal of
    the music profile above 400 Hz, the long-term speech spectrum below 250 Hz, blended in between."""
    lt = np.asarray(speech_prof["ltas"], float)
    lt = lt - lt[VOICE_CORE].mean()
    if not music_prof.get("stems", {}).get("vocals"):
        return lt, "ltass"
    sv = np.asarray(music_prof["stems"]["vocals"]["shape"], float)
    sv = sv - sv[VOICE_CORE].mean()
    w = np.clip(np.log(CENT / 250.0) / np.log(400.0 / 250.0), 0, 1)
    return (1 - w) * lt + w * sv, "studio_vocal"


def _quiet_gaps(mix, gaps):
    """Samples of the quieter half of the gaps between sentences (100 ms blocks): the steady background under the
    talk, without the cheering bursts."""
    blk = int(0.1 * SR)
    idx = np.flatnonzero(gaps[::blk][:len(mix) // blk])
    if len(idx) < 20:
        return None
    lv = np.array([np.mean(mix[i * blk:(i + 1) * blk].astype(np.float64) ** 2) for i in idx])
    keep = idx[lv <= np.median(lv)]
    q = np.zeros(len(mix), bool)
    for i in keep:
        q[i * blk:(i + 1) * blk] = True
    return q


def voice_curve(v, mix, m, target, p, low_limit, cutoff, gaps=None):
    """v2 voice EQ, as the music path's per-stem curve: toward the target, boosts scaled by how much the voice
    dominates the mix in each band (no boosting the crowd), clipped; nothing below the high-pass or above the
    voice's own bandwidth is boosted. When the voice is the mix itself (no usable stems), its share per band is
    estimated as talk minus the quieter half of the gaps (gaps: speech-region samples outside the talker mask)."""
    vs, ms = (v[m], mix[m]) if m.sum() > 3 * SR else (v, mix)
    f, Pv = dsp.psd(vs)
    _, Pm = dsp.psd(ms)
    Bv, Bm = dsp.bands(f, Pv), dsp.bands(f, Pm)
    cur = 10 * np.log10(Bv + 1e-30)
    cur -= cur[VOICE_CORE].mean()
    d = uniform_filter1d(target - cur, 3, mode="nearest")
    share = Bv / (Bm + 1e-30)
    q = _quiet_gaps(mix, gaps) if (gaps is not None and v is mix) else None
    if q is not None and q.sum() > 2 * SR:
        _, Pg = dsp.psd(mix[q])
        share = np.maximum(Bm - dsp.bands(f, Pg), 0) / (Bm + 1e-30)
    conf = np.clip((share - 0.05) / 0.25, 0, 1)
    g = d * p["alpha"]
    g = np.where(g > 0, g * conf, g)
    g = np.clip(g, p["lo_db"], p["hi_db"])
    g[CENT < low_limit] = np.minimum(g[CENT < low_limit], 0)
    if cutoff:
        top = CENT > cutoff - 1000
        if (~top).any():
            g[top] = np.minimum(g[top], g[~top][-1])
    return g


def chain_curve(S, chain):
    """The vocal-stem EQ of the songs: from this render (chain) or from another work folder (match.chain_file)."""
    if S["match"].get("chain_file"):
        with open(S["match"]["chain_file"]) as fh:
            return np.asarray(json.load(fh)["mean"], float), S["match"]["chain_file"]
    if chain is not None:
        return np.asarray(chain["mean"], float), "the songs in this file"
    return None, None


def _fmt(g):
    return " ".join(f"{f}:{np.interp(np.log(f), np.log(CENT), g):+.1f}"
                    for f in (100, 160, 250, 400, 1000, 2000, 3150, 5000, 8000, 12500))


def afftdn(y, nr_db, nf_db):
    if not audio_io.have_ffmpeg() or nr_db <= 0:
        return y
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                        "-af", f"afftdn=nr={nr_db}:nf={nf_db:.0f}:tn=1", "-f", "f32le", "-"],
                       input=np.ascontiguousarray(y, np.float32).tobytes(), capture_output=True, check=True)
    out = np.frombuffer(p.stdout, np.float32).reshape(-1, 2)
    return np.ascontiguousarray(out[:len(y)] if len(out) >= len(y) else np.vstack([out, np.zeros((len(y) - len(out), 2),
                                                                                                  np.float32)]))


def leveler(v, act, hop_s, range_db=6.0, window_s=3.0):
    """Slow gain riding toward the median speech level, computed on speech frames only."""
    h = int(hop_s * SR)
    n = min(len(act), len(v) // h)
    e = (v[:n * h].astype(np.float64) ** 2).mean(axis=1).reshape(n, h).mean(axis=1)
    k = int(window_s / hop_s)
    from scipy.ndimage import uniform_filter1d
    num = np.maximum(uniform_filter1d(e * act[:n], k), 0.0)
    den = uniform_filter1d(act[:n].astype(float), k)
    lvl = 10 * np.log10(num / np.maximum(den, 1e-6) + 1e-20)
    valid = den > 0.2
    if not valid.any():
        return v
    med = np.median(lvl[valid])
    g = np.where(valid, np.clip(med - lvl, -range_db, range_db), 0.0)
    g = uniform_filter1d(g, int(1.0 / hop_s))
    gs = np.repeat(undb(g), h)
    gs = np.concatenate([gs, np.full(len(v) - len(gs), gs[-1] if len(gs) else 1.0)])
    return (v * gs[:, None]).astype(np.float32)


def crowd_taming(y, act, hop_s, cap_db=2.0, floor_db=18.0, gap_db=4.0):
    """Outside speech, hold the crowd's short-term level at (median speech level - cap_db), with at most floor_db of
    reduction and gap_db of gentle downward expansion; smooth release, early attack, and speech frames never move
    by more than 1.5 dB."""
    h = int(hop_s * SR)
    n = min(len(act), len(y) // h)
    mono = y[:n * h].mean(axis=1).astype(np.float64)
    env = np.sqrt(np.convolve((mono ** 2).reshape(n, h).mean(axis=1), np.ones(40) / 40, "same"))
    env_db = 20 * np.log10(env + 1e-9)
    a = act[:n]
    if not a.any() or a.all():
        return y, {"reduced_pct": 0.0}
    sp = float(np.median(env_db[a]))
    g = np.zeros(n)
    g[~a] = np.minimum(np.clip((sp - cap_db) - env_db[~a], -floor_db, 0.0), -gap_db)
    for i in range(1, n):                                    # release toward 0 dB at ~25 dB/s
        if g[i] > g[i - 1]:
            g[i] = min(g[i], g[i - 1] + 0.25)
    for i in range(n - 2, -1, -1):                           # start reducing up to ~120 ms early
        g[i] = min(g[i], g[i + 1] + 1.5)
    g[a] = np.maximum(g[a], -1.5)
    g = np.convolve(g, np.ones(15) / 15, "same")
    gs = np.repeat(undb(g), h)
    gs = np.concatenate([gs, np.full(len(y) - len(gs), gs[-1])])
    return (y * gs[:, None]).astype(np.float32), {"speech_level_db": round(sp, 1),
                                                  "reduced_pct": round(float(np.mean(g < -1) * 100), 1),
                                                  "max_reduction_db": round(float(g.min()), 1)}


def run(x, st, plan, speech_mask, act, hop=0.01, log=print, work=None, chain=None):
    S = plan["speech"]
    mode = mode_of(S)
    sprof = profiles.load(plan["profiles"]["speech"])
    src = S["source"]
    voice_is_mix = not (src["value"] == "stems" and st is not None)
    if not voice_is_mix:
        v = st["vocals"].copy()
        amb = dsp.highpass(x - st["vocals"], 90) * float(undb(src["ambience_db"]))
    else:
        v = x.copy()
        amb = np.zeros_like(x)
    h = int(hop * SR)
    nfr = min(len(act), len(speech_mask) // h)
    act = act[:nfr] & speech_mask[:nfr * h].reshape(nfr, h).any(axis=1)   # singing inside songs is not speech
    m = np.repeat(act, h)
    m = np.concatenate([m, np.zeros(len(v) - len(m), bool)])[:len(v)] & speech_mask
    info = {"activity": act, "hop": hop, "mode": mode}
    log(f"[speech] mode {mode}")

    # 1. de-limit (v2): the talk of a broadcast went through the same limiter as the songs
    dl = S.get("delimit", {})
    if mode == "v2" and dl.get("enabled"):
        cache = os.path.join(work, "delimit_gain.npy") if work else None
        if cache and os.path.exists(cache):
            g = np.load(cache)[:len(x)]
        else:
            log("[speech] de-limiter inference ...")
            g = delimit.gain_curve(x)
            if cache:
                np.save(cache, g)
        gm = delimit.modulation(g, dl["depth"])[:, None]
        v, amb = v * gm, amb * gm
        log(f"[speech] de-limit depth {dl['depth']}")

    # 2. stationary denoise and high-pass
    if S["denoise"]["enabled"]:
        lv, _ = measure.rms_db(v, 0.1, 0.05)
        nf = float(np.clip(np.percentile(lv, 10), -80, -30))
        v = afftdn(v, S["denoise"]["nr_db"], nf)
        log(f"[speech] afftdn nr {S['denoise']['nr_db']} dB, noise floor {nf:.0f} dB")
    hp = S["highpass_hz"]["value"]
    v = dsp.highpass(v, hp)

    # 3. voice EQ
    if S["match"]["enabled"]:
        p = S["match"]
        if mode == "gentle":
            cur = dsp.ltas(v[m] if m.sum() > 3 * SR else v)
            g = dsp.match_curve(cur, np.asarray(sprof["ltas"]), GENTLE["alpha"], GENTLE["lo_db"], GENTLE["hi_db"],
                                cutoff=p["cutoff_hz"], low_limit=hp * 1.2)
            src_txt = "half-way to the speech spectrum"
        else:
            g, src_txt = None, None
            if p.get("target") == "song_vocal":
                g, where = chain_curve(S, chain)
                if g is not None:
                    src_txt = f"the vocal EQ of {where}"
                    g = np.where(CENT < hp * 1.2, np.minimum(g, 0), g)
            if g is None:
                tgt, kind = voice_target(profiles.load(plan["profiles"]["music"]), sprof)
                mixref = v if voice_is_mix else x
                g = voice_curve(v, mixref, m, tgt, p, hp * 1.2, p["cutoff_hz"], gaps=speech_mask & ~m)
                src_txt = f"{p['alpha']:.0%} toward the {kind.replace('_', ' ')} target"
        v = dsp.apply_fir(v, dsp.fir_from_curve(g))
        info["voice_eq_db"] = [round(float(a), 2) for a in g]
        log(f"[speech] voice EQ ({src_txt}): {_fmt(g)}")

    # 4. de-ess, compression, leveler
    if S["deess"]["enabled"]:
        v, a = dsp.deess(v, pct=S["deess"]["pct"], max_cut=S["deess"]["max_cut_db"])
        log(f"[speech] de-ess {a}")
    c = S["compress"]
    r, at, rl, pc = GENTLE["compress"] if mode == "gentle" else (c["ratio"], c["att_ms"], c["rel_ms"], c["pct"])
    v, a = dsp.compress(v, r, at, rl, pct=pc)
    log(f"[speech] compress {r}:1 {at}/{rl} ms {a}")
    if S["leveler"]["enabled"]:
        v = leveler(v, act, hop, S["leveler"]["range_db"], S["leveler"]["window_s"])

    # 5. room underneath, crowd held down between sentences
    y = v + amb
    if S["crowd"]["enabled"]:
        c = S["crowd"]
        y, a = crowd_taming(y, act, hop, c["cap_db"], c["floor_db"], c["gap_db"])
        log(f"[speech] crowd taming {a}")

    # 6. band extension above a codec wall (v2), gentler than for music
    be = S.get("band_ext", {})
    if mode == "v2" and be.get("enabled") and m.sum() > 3 * SR:
        fc = be["cutoff_hz"]
        tgt = dsp.band_rel(y[m], fc - 2500, fc - 300) - 6.0
        y, a = dsp.hf_extend(y, fc, tgt - be.get("below_ref_db", 3.0), mask=m)
        log(f"[speech] band extension above {fc} Hz: {a}")
    return y.astype(np.float32), info
