"""The tonal reference for a song: the studio version of the very song when it can be found online, else the bundled
default profile.

The studio master of the song is what "like the album" means, and it is the best target the skill knows. It used to
depend on the assistant knowing to build it with `profile --itunes-song`; most did not. Now `analyze` / `auto` look it
up themselves from the artist and title (the file's tags, or --artist / --title), whenever no profile was chosen and
the network allows. `analysis.run` then checks that the preview's passage is found in the recording (centred chroma
correlation >= 0.35, standing out in its key; `profiles.locate`) and falls back to the default otherwise: a wrong
song or another arrangement is a worse target than none."""
import json, os, re

from . import profiles

DEFAULT = "jrock_band"
NOT_SONGS = re.compile(r"^(mc|ｍｃ|talk|トーク|se|opening|intro|outro|encore|アンコール)\s*\d*$", re.I)


def identity(tags=None, artist=None, title=None):
    """(artist, title) from --artist / --title, else from the container tags, with track numbers and live markers
    removed from the title."""
    t = {str(k).lower(): v for k, v in (tags or {}).items()}
    a = (artist or t.get("artist") or t.get("album_artist") or t.get("album artist") or "").strip()
    s = (title or t.get("title") or "").strip()
    s = re.sub(r"^\d{1,3}\s*[.\-_)]\s*(?=\D)", "", s)                     # "18. Title", "18 - Title"
    s = re.sub(r"\s*[(\[（【]\s*(live|ライブ|ver\.? *live|live ver(sion|\.)?)[^)\]）】]*[)\]）】]\s*$", "", s, flags=re.I)
    s = re.sub(r"\s+-\s+live\b.*$", "", s, flags=re.I)
    return a, s.strip()


def stores(text, first=None):
    """iTunes store countries to try, from the script of the artist and title."""
    if re.search(r"[぀-ヿ]", text):                                 # kana: Japanese
        order = ["jp", "us"]
    elif re.search(r"[가-힯]", text):                               # hangul
        order = ["kr", "us"]
    elif re.search(r"[一-鿿]", text):                               # ideographs only: Japanese or Chinese
        order = ["jp", "tw", "hk", "cn", "us"]
    else:
        order = ["us", "gb", "jp"]
    return [first] + [c for c in order if c != first] if first else order


def find(tags, work, artist=None, title=None, country=None, lookup=True, with_stems=True, log=print):
    """Returns (profile, info). profile is a JSON path in WORK/reference (the studio version's preview) or DEFAULT;
    info records what was tried and why the result was chosen. Cached in the work folder."""
    a, s = identity(tags, artist, title)
    info = {"auto": True, "artist": a or None, "title": s or None, "used": DEFAULT}
    if not lookup:
        info["why"] = "online lookup turned off (--no-lookup): the bundled default profile"
        return DEFAULT, info
    if not (a and s):
        info["why"] = ("no artist and title (tags, or --artist / --title), so the studio version could not be looked "
                       "up: the bundled default profile")
        return DEFAULT, info
    if NOT_SONGS.match(s):
        info["why"] = f"'{s}' is not a song title: the bundled default profile"
        return DEFAULT, info
    folder = os.path.join(work, "reference")
    out, meta = os.path.join(folder, "song.json"), os.path.join(folder, "lookup.json")
    if os.path.exists(out) and os.path.exists(meta):
        with open(meta) as fh:
            old = json.load(fh)
        if old.get("artist") == a and old.get("title") == s:
            log(f"[reference] studio version of '{s}' by {a}: cached in {folder}")
            return out, old
    log(f"[reference] looking up the studio version of '{s}' by {a} (iTunes previews) ...")
    tried = []
    for c in stores(f"{a} {s}", country):
        got = {}
        try:
            files = profiles.fetch_itunes(a, os.path.join(folder, "previews"), country=c, song=s, info=got)
        except SystemExit:                                                  # the song is not in this store
            tried.append(c)
            continue
        except Exception as e:                                              # noqa: BLE001  offline, blocked, timeout
            info["why"] = (f"no network access to the iTunes Search API ({type(e).__name__}): the bundled default "
                           f"profile. Ask the listener for the studio version of the song and build the reference "
                           f"with `profile --files STUDIO --song`")
            log(f"[reference] {info['why']}")
            return DEFAULT, info
        if not files:
            tried.append(c)
            continue
        tracks = got.get("tracks", [])
        src = "; ".join(f"'{t['track']}' from '{t['collection']}'" for t in tracks) or f"{len(files)} previews"
        prof = profiles.build(files, f"studio: {s}", "music", os.path.join(folder, "build"), with_stems=with_stems,
                              source=f"official iTunes preview ({c} store) of {src} by {a}", song=True)
        profiles.save(prof, out)
        info.update(used=out, store=c, tracks=tracks,
                    why=f"studio version found online: {src} (iTunes {c} store)")
        with open(meta, "w") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=1)
        log(f"[reference] {info['why']} -> {out}")
        return out, info
    info["why"] = (f"no studio version of '{s}' by {a} in the iTunes stores {', '.join(tried)}: the bundled default "
                   f"profile. The listener can supply the studio track (`profile --files STUDIO --song`)")
    log(f"[reference] {info['why']}")
    return DEFAULT, info
