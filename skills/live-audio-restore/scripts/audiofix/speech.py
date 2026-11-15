"""Speech (MC) path: voice from the vocal stem with the room underneath, gentle stationary denoise, high-pass,
half-way match to the long-term speech spectrum, de-ess, compression and a slow leveler, then crowd taming
between sentences (never more than 1.5 dB on speech). No neural denoisers: on PA'd, reverberant live speech
they remove speech along with the noise."""
import subprocess
import numpy as np

from . import SR, audio_io, dsp, measure, profiles  # noqa: F401
from .dsp import undb


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


def run(x, st, plan, speech_mask, act, hop=0.01, log=print):
    S = plan["speech"]
    prof = profiles.load(plan["profiles"]["speech"])
    src = S["source"]
    if src["value"] == "stems" and st is not None:
        v = st["vocals"].copy()
        amb = dsp.highpass(x - st["vocals"], 90) * float(undb(src["ambience_db"]))
    else:
        v = x.copy()
        amb = np.zeros_like(x)
    h = int(hop * SR)
    nfr = min(len(act), len(speech_mask) // h)
    act = act[:nfr] & speech_mask[:nfr * h].reshape(nfr, h).any(axis=1)   # singing inside songs is not speech
    if S["denoise"]["enabled"]:
        lv, _ = measure.rms_db(v, 0.1, 0.05)
        nf = float(np.clip(np.percentile(lv, 10), -80, -30))
        v = afftdn(v, S["denoise"]["nr_db"], nf)
        log(f"[speech] afftdn nr {S['denoise']['nr_db']} dB, noise floor {nf:.0f} dB")
    v = dsp.highpass(v, S["highpass_hz"]["value"])
    if S["match"]["enabled"]:
        p = S["match"]
        m = np.repeat(act, h)
        m = np.concatenate([m, np.zeros(len(v) - len(m), bool)]) & speech_mask
        cur = dsp.ltas(v[m] if m.sum() > 3 * SR else v)
        g = dsp.match_curve(cur, np.asarray(prof["ltas"]), p["alpha"], p["lo_db"], p["hi_db"], cutoff=p["cutoff_hz"],
                            low_limit=S["highpass_hz"]["value"] * 1.2)
        v = dsp.apply_fir(v, dsp.fir_from_curve(g))
        log("[speech] voice EQ " + " ".join(f"{f}:{np.interp(np.log(f), np.log(dsp.CENT), g):+.1f}"
                                           for f in (100, 200, 315, 500, 1000, 2000, 3150, 5000, 8000)))
    if S["deess"]["enabled"]:
        v, a = dsp.deess(v, pct=S["deess"]["pct"], max_cut=S["deess"]["max_cut_db"])
        log(f"[speech] de-ess {a}")
    c = S["compress"]
    v, a = dsp.compress(v, c["ratio"], c["att_ms"], c["rel_ms"], pct=c["pct"])
    log(f"[speech] compress {c['ratio']}:1 {a}")
    if S["leveler"]["enabled"]:
        v = leveler(v, act, hop, S["leveler"]["range_db"], S["leveler"]["window_s"])
    y = v + amb
    if S["crowd"]["enabled"]:
        c = S["crowd"]
        y, a = crowd_taming(y, act, hop, c["cap_db"], c["floor_db"], c["gap_db"])
        log(f"[speech] crowd taming {a}")
    return y.astype(np.float32), {"activity": act, "hop": hop}
