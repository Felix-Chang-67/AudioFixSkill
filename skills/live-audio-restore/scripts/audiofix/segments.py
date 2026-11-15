"""Content map: which stretches are music, speech (MC) or gaps (applause, silence, crowd), and where the
music should be split into sections that get their own tonal treatment (e.g. a change of broadcast feed)."""
import numpy as np
from scipy.ndimage import median_filter

from . import SR
from .dsp import bandpass, db

CODES = {"music": 0, "speech": 1, "gap": 2}
NAMES = {v: k for k, v in CODES.items()}


def _power(y, w, h):
    p = (y.astype(np.float64) ** 2).mean(axis=1)
    c = np.concatenate([[0.0], np.cumsum(p)])
    st = np.arange(0, max(1, len(p) - w + 1), h)
    return (c[st + w] - c[st]) / w, st


def _pauses(vocals, starts, w):
    """Share of 10 ms frames in each window that sit 20 dB below the window's loudest: speech pauses between
    phrases, singing is continuous (median ~0.05 sung vs ~0.45 spoken on live material)."""
    vb = bandpass(vocals, 300, 3400).mean(axis=1)
    h = int(0.01 * SR)
    n = len(vb) // h
    edb = db(np.sqrt((vb[:n * h].astype(np.float64) ** 2).reshape(n, h).mean(axis=1)) + 1e-9)
    out = np.zeros(len(starts))
    W = w // h
    for k, s in enumerate(starts):
        e = edb[s // h:s // h + W]
        out[k] = float(np.mean(e < e.max() - 20)) if len(e) and e.max() > -60 else 1.0
    return out


def classify(mix, stems=None, content="auto", win_s=3.0, hop_s=1.0, word_density=None):
    """Per-second labels and merged segments [(start_s, end_s, label)].

    music  : drums+bass carry a real share of the energy (median over 9 s), or nothing points to speech
    speech : little rhythm section and either (a) Whisper finds talk-like word density (>= 0.6 words/s over 9 s,
             works even when separation fails), or (b) a vocal is present and pauses like talking (>25 % of frames)
    gap    : near silence (35 dB under the loud parts) or crowd noise without voice or band"""
    n = len(mix)
    w, h = int(win_s * SR), int(hop_s * SR)
    pm, starts = _power(mix, w, h)
    lvl = 10 * np.log10(pm + 1e-20)
    ref = np.percentile(lvl, 90)
    lab = np.full(len(pm), CODES["music"])
    feats = {}
    if content in ("music", "speech"):
        lab[:] = CODES[content]
    elif stems is not None:
        pv, _ = _power(stems["vocals"], w, h)
        pd, _ = _power(stems["drums"], w, h)
        pb, _ = _power(stems["bass"], w, h)
        rhythm = median_filter((pd + pb) / (pm + 1e-20), size=9, mode="nearest")
        vshare = median_filter(pv / (pm + 1e-20), size=9, mode="nearest")
        pause = median_filter(_pauses(stems["vocals"], starts, w), size=9, mode="nearest")
        speech = (rhythm < 0.08) & (pause > 0.25) & (vshare > 0.15)
        if word_density is not None:
            wd = median_filter(np.asarray(word_density, float)[:len(rhythm)], size=9, mode="nearest")
            wd = np.pad(wd, (0, max(0, len(rhythm) - len(wd))))
            speech |= (rhythm < 0.08) & (wd >= 0.6)
        crowd = (rhythm < 0.05) & (vshare < 0.15) & ~speech
        lab[speech] = CODES["speech"]
        lab[crowd] = CODES["gap"]
        feats = {"rhythm": rhythm, "vocal_share": vshare, "pause": pause,
                 "default": "speech" if np.median(rhythm) < 0.03 else "music"}
    lab[lvl < ref - 35] = CODES["gap"]
    lab = median_filter(lab, size=5, mode="nearest")
    segs = _merge(_runs(lab, hop_s, n / SR), {"music": 20.0, "speech": 10.0, "gap": 2.0})
    return segs, feats


def _runs(lab, hop_s, total):
    out = []
    a = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[a]:
            out.append([a * hop_s, min(total, i * hop_s if i < len(lab) else total), NAMES[int(lab[a])]])
            a = i
    if out:
        out[0][0] = 0.0
    return out


def _merge(segs, min_len):
    """Absorb too-short segments into their longer neighbour, then join equal neighbours."""
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b, c) in enumerate(segs):
            if b - a < min_len[c]:
                if i == 0:
                    j = 1
                elif i == len(segs) - 1:
                    j = i - 1
                else:
                    j = i - 1 if (segs[i - 1][1] - segs[i - 1][0]) >= (segs[i + 1][1] - segs[i + 1][0]) else i + 1
                lo, hi = min(i, j), max(i, j)
                keep = segs[j][2]
                segs[lo:hi + 1] = [[segs[lo][0], segs[hi][1], keep]]
                changed = True
                break
    out = []
    for s in segs:
        if out and out[-1][2] == s[2]:
            out[-1][1] = s[1]
        else:
            out.append(list(s))
    return out


def assign_gaps(segs, default="music"):
    """Processing classes only know music and speech: a gap takes the class of its neighbours (split in the middle
    when they differ). With nothing but gaps, `default` decides (speech when there is no rhythm section).
    Returns [(a, b, 'music'|'speech')], merged."""
    real = [s for s in segs if s[2] != "gap"]
    if not real:
        return [[0.0, segs[-1][1], default]]
    out = []
    for i, (a, b, c) in enumerate(segs):
        if c != "gap":
            out.append([a, b, c]); continue
        prev = next((segs[j][2] for j in range(i - 1, -1, -1) if segs[j][2] != "gap"), None)
        nxt = next((segs[j][2] for j in range(i + 1, len(segs)) if segs[j][2] != "gap"), None)
        if prev and nxt and prev != nxt:
            m = (a + b) / 2
            out += [[a, m, prev], [m, b, nxt]]
        else:
            out.append([a, b, prev or nxt])
    merged = []
    for s in out:
        if merged and merged[-1][2] == s[2]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def find_cuts(mix, a, b, win_s=2.0, ctx_s=20.0, min_gap_s=30.0, threshold=1.2):
    """Abrupt changes of the recording chain inside a music stretch (e.g. a switch of broadcast feed), seen as a
    jump of the stereo image: side-minus-mid level and mid-band L-R correlation, medians over 20 s either side.
    Tonal shape is deliberately not used: it also changes with the arrangement, which must not be equalised away."""
    if b - a < 2 * min_gap_s:
        return []
    w = int(win_s * SR)
    i0, i1 = int(a * SR), int(b * SR)
    t, sm, corr, lvl = [], [], [], []
    mid_band = bandpass(mix[i0:i1], 300, 3000)
    for i in range(i0, i1 - w + 1, w):
        seg = mix[i:i + w]
        s = (seg[:, 0] - seg[:, 1]) / 2; m = (seg[:, 0] + seg[:, 1]) / 2
        sm.append(db(np.std(s) / (np.std(m) + 1e-12)))
        mb = mid_band[i - i0:i - i0 + w]
        corr.append(np.corrcoef(mb[:, 0], mb[:, 1])[0, 1] if np.std(mb) > 1e-6 else 1.0)
        lvl.append(db(np.sqrt(np.mean(m ** 2)) + 1e-12))
        t.append(i / SR)
    t, sm, corr, lvl = map(np.asarray, (t, sm, corr, lvl))
    k = int(ctx_s / win_s)
    nov = np.zeros(len(t))
    for j in range(k, len(t) - k):
        b4, af = slice(j - k, j), slice(j, j + k)
        nov[j] = abs(np.median(sm[af]) - np.median(sm[b4])) / 4 + abs(np.median(corr[af]) - np.median(corr[b4])) / 0.5
    cuts = []
    for j in np.argsort(-nov):
        if nov[j] < threshold:
            break
        tj = t[j]
        if tj - a < min_gap_s or b - tj < min_gap_s or any(abs(tj - c) < min_gap_s for c in cuts):
            continue
        near = np.flatnonzero(np.abs(t - tj) <= 3.0)
        cuts.append(float(t[near[np.argmin(lvl[near])]] + win_s / 2))
    return sorted(cuts)


def sections(proc_segs, mix, cut_threshold=1.2):
    """Music sections for per-section tonal treatment: music stretches, split at abrupt changes."""
    out = []
    for a, b, c in proc_segs:
        if c != "music":
            continue
        cuts = find_cuts(mix, a, b, threshold=cut_threshold)
        edges = [a] + cuts + [b]
        out += [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]
    return out
