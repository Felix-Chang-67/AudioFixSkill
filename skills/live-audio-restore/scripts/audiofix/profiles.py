"""Reference profiles: a numeric description of what good material of this kind sounds like.

A profile holds no audio, only statistics:
  ltas       long-term spectrum on the 1/6-octave grid, dB re the 0.5-2 kHz mean
  hf_rel_db  power of 1 kHz bands from 8 to 20 kHz re the 0.5-2 kHz power (sets band-extension level)
  stems      per stem (music only): spectral shape normalised at the stem's core band, and its share of the mix
Build one from the artist's own studio recordings (best), from official preview clips, or use a bundled one.
"""
import glob, json, os, urllib.parse, urllib.request
import numpy as np

from . import SR
from .dsp import CENT, band_rel, bands, ltas, psd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "profiles"))
CORE = {"vocals": (300, 1000), "other": (300, 1000), "drums": (100, 5000), "bass": (60, 250)}
HF_BANDS = list(range(8000, 20000, 1000))


def stem_shape(y, stem):
    f, P = psd(y)
    B = 10 * np.log10(bands(f, P) + 1e-30)
    lo, hi = CORE[stem]
    sel = (CENT >= lo) & (CENT <= hi)
    return B - B[sel].mean()


def hf_rel(y):
    return {str(b): round(band_rel(y, b, b + 1000), 2) for b in HF_BANDS}


def hf_target(profile, lo, hi):
    """Expected power of [lo, hi) re 0.5-2 kHz according to the profile (power average of its 1 kHz bands)."""
    rel = profile.get("hf_rel_db") or {}
    vals = [10 ** (v / 10) for b, v in rel.items() if int(b) + 1000 > lo and int(b) < hi]
    return float(10 * np.log10(np.mean(vals))) if vals else None


def load(name_or_path):
    p = name_or_path
    if not os.path.exists(p):
        p = os.path.join(ASSETS, f"{name_or_path}.json")
    with open(p) as fh:
        prof = json.load(fh)
    prof["ltas"] = np.asarray(prof["ltas"])
    for s in (prof.get("stems") or {}).values():
        s["shape"] = np.asarray(s["shape"])
    return prof


def save(prof, path):
    out = dict(prof)
    out["ltas"] = [round(float(v), 3) for v in prof["ltas"]]
    if prof.get("stems"):
        out["stems"] = {k: {"shape": [round(float(x), 3) for x in v["shape"]], "ratio_db": round(v["ratio_db"], 2)}
                        for k, v in prof["stems"].items()}
    out["cent"] = [round(float(c), 2) for c in CENT]
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


def build(files, name, kind="music", work=None, with_stems=True, source=""):
    """Median statistics over reference recordings (any format ffmpeg can read)."""
    from . import audio_io, loudness, stems as S
    lt, hf, lu, plr = [], [], [], []
    shapes = {s: [] for s in S.STEMS}
    ratios = {s: [] for s in S.STEMS}
    use_stems = with_stems and kind == "music" and S.available()
    for p in files:
        y = audio_io.load(p)
        if len(y) < 5 * SR:
            continue
        lt.append(ltas(y)); hf.append(hf_rel(y))
        lu.append(loudness.integrated(y)); plr.append(loudness.true_peak(y) - lu[-1])
        if use_stems:
            wav = os.path.join(work, "refs", os.path.splitext(os.path.basename(p))[0] + ".wav")
            os.makedirs(os.path.dirname(wav), exist_ok=True)
            audio_io.save_wav(wav, y, bits=32)
            st = S.load(S.separate(wav, os.path.join(work, "refs")), len(y))
            mix = sum(st.values())
            pm = np.mean(mix.astype(np.float64) ** 2)
            for s in S.STEMS:
                shapes[s].append(stem_shape(st[s], s))
                ratios[s].append(10 * np.log10(np.mean(st[s].astype(np.float64) ** 2) / pm + 1e-30))
    if not lt:
        raise SystemExit("no usable reference audio (need at least 5 s per file)")
    prof = {"name": name, "kind": kind, "source": source, "count": len(lt),
            "ltas": np.median(np.array(lt), axis=0),
            "hf_rel_db": {b: round(float(np.median([h[b] for h in hf])), 2) for b in hf[0]},
            "loudness": {"lufs_median": round(float(np.median(lu)), 1), "plr_median": round(float(np.median(plr)), 1)}}
    if use_stems and shapes["vocals"]:
        prof["stems"] = {s: {"shape": np.median(np.array(shapes[s]), axis=0), "ratio_db": float(np.median(ratios[s]))}
                         for s in S.STEMS}
    return prof


def fetch_itunes(artist, out_dir, limit=30, country="us"):
    """Download official 30 s preview clips of an artist from the public iTunes Search API (tonal reference only)."""
    os.makedirs(out_dir, exist_ok=True)
    q = urllib.parse.urlencode({"term": artist, "entity": "song", "limit": 200, "country": country})
    data = json.load(urllib.request.urlopen(f"https://itunes.apple.com/search?{q}", timeout=30))
    res = [t for t in data.get("results", []) if t.get("previewUrl")]
    match = [t for t in res if artist.lower() in (t.get("artistName") or "").lower()]
    if not match and res:                          # e.g. a romanised name in a store that lists the artist in kana
        from collections import Counter
        top = Counter(t["artistId"] for t in res[:25]).most_common(1)[0][0]
        match = [t for t in res if t["artistId"] == top]
    if match:
        print(f"iTunes: using previews of '{match[0].get('artistName')}' (artist id {match[0].get('artistId')})")
    got = []
    for t in match:
        fn = os.path.join(out_dir, f"{t['trackId']}.m4a")
        if not os.path.exists(fn):
            urllib.request.urlretrieve(t["previewUrl"], fn)
        got.append(fn)
        if len(got) >= limit:
            break
    return got


# Long-term average speech spectrum, 1/3-octave band levels (dB SPL at 65 dB overall), combined male and female
# talkers, approximately after Byrne et al. (1994), J. Acoust. Soc. Am. 96(4). Used only as a soft target.
LTASS = {63: 38.6, 80: 43.5, 100: 54.4, 125: 57.7, 160: 56.8, 200: 60.2, 250: 60.3, 315: 59.0, 400: 62.1,
         500: 62.1, 630: 60.5, 800: 56.8, 1000: 53.7, 1250: 53.0, 1600: 52.0, 2000: 48.7, 2500: 48.1, 3150: 46.8,
         4000: 45.6, 5000: 44.5, 6300: 44.3, 8000: 43.7, 10000: 43.4, 12500: 41.3, 16000: 40.7}


def speech_ltass():
    """LTASS band levels -> power-density shape on our grid (subtract 10*log10 of the 1/3-octave bandwidth)."""
    f = np.array(sorted(LTASS)); lv = np.array([LTASS[k] for k in f])
    dens = lv - 10 * np.log10(0.2316 * f)
    cur = np.interp(np.log(CENT), np.log(f), dens)
    sel = (CENT >= 500) & (CENT <= 2000)
    return {"name": "speech_ltass", "kind": "speech", "count": 1,
            "source": "Long-term average speech spectrum, approx. Byrne et al. 1994 (combined talkers)",
            "ltas": cur - cur[sel].mean(), "hf_rel_db": {}}


def available_profiles():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ASSETS, "*.json")))
