"""Command line: check | profile | analyze | restore | verify | auto. Run `python3 scripts/audiofix.py -h`."""
import argparse, json, os, sys

from . import (__version__, analysis, audio_io, delimit, pipeline, plan as PL, profiles, reference, report, review,
               stems, verify)


class Log:
    def __init__(self, work=None):
        self.fh = open(os.path.join(work, "log.txt"), "a") if work else None

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if self.fh:
            self.fh.write(s + "\n"); self.fh.flush()


def cmd_check(a):
    cap = analysis.capabilities()
    print(json.dumps(cap, indent=1))
    missing = {"ffmpeg": "decode m4a/mp3/video audio, AAC export, stationary denoise, declip: install ffmpeg",
               "stems": "stem separation (needed for the full music and MC paths): pip install demucs torch",
               "delimiter": "neural de-limiting: run scripts/setup.sh (clones De-limiter, installs asteroid deps)",
               "numba": "speed only: pip install numba (vectorised approximations are used without it)",
               "asr": "word-accurate MC talker timing and the intelligibility check: pip install faster-whisper "
                      "(the model downloads on first use)",
               "codec_repair": "Apollo codec repair (natural highs on low-bitrate rips): run scripts/setup.sh "
                               "(clones Apollo; the checkpoint downloads on first use)"}
    for k, ok in cap.items():
        print(f"{'ok ' if ok else 'MISSING'} {k}" + ("" if ok else f"  -> {missing.get(k, '')}"))
    from . import device
    print(f"compute device for Apollo and demucs: {device.describe()}"
          + ("" if device.pick() != "cpu" else "  (codec repair: about 6x real time; set AUDIOFIX_DEVICE to force one)"))
    print("profiles:", profiles.available_profiles())


def cmd_profile(a):
    work = a.work or os.path.join(os.path.dirname(a.out) or ".", "profile_work")
    os.makedirs(work, exist_ok=True)
    files = list(a.files or [])
    src = a.source or ""
    if a.itunes_artist:
        files += profiles.fetch_itunes(a.itunes_artist, os.path.join(work, "previews"), a.limit, a.country,
                                        song=a.itunes_song)
        src = src or (f"official iTunes preview of '{a.itunes_song}' by {a.itunes_artist} ({len(files)} clips)"
                      if a.itunes_song else f"official iTunes preview clips of {a.itunes_artist} ({len(files)} clips)")
    if a.kind == "speech" and not files:
        prof = profiles.speech_ltass()
    else:
        prof = profiles.build(files, a.name, a.kind, work, with_stems=not a.no_stems, source=src or f"{len(files)} files",
                              song=bool(a.itunes_song or a.song))
    profiles.save(prof, a.out)
    print(f"profile '{a.name}' from {prof['count']} references -> {a.out}")


def _analyze(a, log):
    if getattr(a, "language", None):
        os.environ["AUDIOFIX_ASR_LANG"] = a.language
    if getattr(a, "asr_model", None):
        os.environ["AUDIOFIX_ASR_MODEL"] = a.asr_model
    D = analysis.run(a.input, a.work, a.content, a.music_profile, a.speech_profile, not a.no_stems, log,
                     codec_repair=getattr(a, "codec_repair", "auto"),
                     lookup={"artist": a.artist, "title": a.title, "country": a.country, "lookup": not a.no_lookup})
    P = PL.build(D)
    PL.save(P, os.path.join(a.work, "plan.json"))
    with open(os.path.join(a.work, "diagnosis.md"), "w") as fh:
        fh.write(report.diagnosis_md(D, P))
    try:
        report.overview_png(audio_io.load(os.path.join(a.work, "input.wav")), D, os.path.join(a.work, "overview.png"))
    except Exception as e:                                   # noqa: BLE001
        log(f"[analyze] overview plot skipped: {e}")
    log(f"[analyze] wrote diagnosis.md, plan.json, overview.png in {a.work}")
    items = review.material(D)
    with open(os.path.join(a.work, "diagnosis.md"), "a") as fh:
        fh.write("\n## Review before rendering\n\n" + ("".join(f"- **{i['kind']}**: {i['text']}\n" for i in items)
                                                       or "Nothing unusual found.\n"))
    for i in items:
        log(f"[review] {i['kind']}: {i['text']}")
    if items:
        log("[next] the [review] lines above describe this recording: read them before rendering. Edit the plan "
            "only for a measured reason (AGENTS.md 5) and keep each edit's reason for the delivery")
    hint = _reference_hint(P)
    if hint:
        log(hint)
    return D, P


def _reference_hint(P):
    """A [next] line when a song was matched to the generic profile: its own studio version is the better target."""
    ref = P.get("reference") or {}
    if not P.get("music") or not ref.get("auto") or ref.get("used") != reference.DEFAULT:
        return None
    return ("[next] the tone target was the bundled generic profile (" + ref.get("why", "") + "). The studio "
            "version of this very song is the best target: rerun with --artist and --title if the tags lack them, "
            "or ask the listener for the studio track and run `profile --name S --files STUDIO --song --out S.json`, "
            "then `auto INPUT --work WORK --music-profile S.json`")


KEEP = ("restored.wav", "restored.m4a", "AB.wav", "compare.png", "verification.md", "verification.json", "master.json",
        "steps.json", "review.md", "plan_used.json")


def _keep_previous(work, log):
    """The last render moves to previous/, so every change can be compared with what came before."""
    if not os.path.exists(os.path.join(work, "restored.wav")):
        return
    prev = os.path.join(work, "previous")
    os.makedirs(prev, exist_ok=True)
    for f in KEEP:
        if os.path.exists(os.path.join(work, f)):
            os.replace(os.path.join(work, f), os.path.join(prev, f))
    log(f"[restore] the previous render is kept in {prev}")


def _restore(a, log):
    P = PL.load(a.plan or os.path.join(a.work, "plan.json"))
    P = PL.override(P, a.set)
    _keep_previous(a.work, log)
    PL.save(P, os.path.join(a.work, "plan_used.json"))
    out, rep = pipeline.restore(a.work, P, log)
    log(f"[restore] wrote {out} ({rep['seconds']} s)")
    return P, rep


def _verify(a, P, rep, log):
    V = verify.run(a.work, P, log, make_plots=not getattr(a, "no_plots", False))
    with open(os.path.join(a.work, "verification.md"), "w") as fh:
        fh.write(report.verification_md(V, P, rep))
    log(f"[verify] wrote verification.md, AB.wav, compare.png in {a.work}")
    warns = [c["check"] for c in V.get("checks", []) if not c["pass"]]
    w = a.work
    D = review.load_json(os.path.join(w, "diagnosis.json")) or {}
    prev = {"plan": review.load_json(os.path.join(w, "previous", "plan_used.json")),
            "verification": review.load_json(os.path.join(w, "previous", "verification.json"))}
    items = review.material(D) + review.render(D, P, V, review.load_json(os.path.join(w, "steps.json")),
                                               review.load_json(os.path.join(w, "plan.json")),
                                               prev if prev["plan"] else None)
    review.write(os.path.join(w, "review.md"), items, "Review before delivering")
    for i in items:
        log(f"[review] {i['kind']}: {i['text']}")
    log("[next] " + (f"{len(warns)} WARN: {'; '.join(warns)}. Before delivering, act on the cause the [review] lines "
                     f"name (SKILL.md step 6), then re-render" if warns else "all checks PASS"))
    if items:
        log(f"[next] review.md has {len(items)} item(s). PASS is not the end: answer every [review] line in the "
            f"delivery (what it is, what was done, what the listener may want)")
    hint = _reference_hint(P)
    if hint:
        log(hint)
    log(f"[next] deliver {os.path.join(a.work, 'restored.wav')}, restored.m4a and AB.wav (the listener starts with "
        f"AB.wav); quote before/after numbers from verification.md and every plan edit you made, with its reason")
    return V


def _warns(V):
    return sum(not c["pass"] for c in V.get("checks", []))


def _swap_back(work, log):
    """Put the render before the autofix back in place; the rejected one goes to autofix_rejected/."""
    rej = os.path.join(work, "autofix_rejected")
    os.makedirs(rej, exist_ok=True)
    for f in KEEP:
        cur, old = os.path.join(work, f), os.path.join(work, "previous", f)
        if os.path.exists(old):
            if os.path.exists(cur):
                os.replace(cur, os.path.join(rej, f))
            os.replace(old, cur)
    log(f"[autofix] kept the render before the fix; the fixed one is in {rej}")


def _autofix(a, V, P, log):
    """WARNs the step ledger traces to a step with a safe remedy are fixed once, automatically: the same input gives
    the same result whichever assistant runs it. Kept only if the WARN count goes down."""
    if getattr(a, "no_autofix", False) or not V.get("checks"):
        return V
    explicit = [s.split("=", 1)[0] for s in (a.set or [])]
    fixes = review.autofix(V, P, review.load_json(os.path.join(a.work, "steps.json")), explicit)
    if not fixes:
        return V
    sets = []
    for fx, why in fixes:
        sets += [f"{k}={json.dumps(v)}" for k, v in fx.items()]
        log(f"[autofix] {why}: " + ", ".join(f"{k}={v}" for k, v in fx.items()) + "; re-rendering once")
    b = argparse.Namespace(**vars(a))
    b.plan, b.set = os.path.join(a.work, "plan_used.json"), sets
    P2, rep2 = _restore(b, log)
    V2 = _verify(b, P2, rep2, log)
    if _warns(V2) >= _warns(V):
        log(f"[autofix] WARNs {_warns(V)} -> {_warns(V2)}: the fix did not help")
        _swap_back(a.work, log)
        return V
    log(f"[autofix] WARNs {_warns(V)} -> {_warns(V2)}. Name this edit and its reason in the delivery")
    return V2


def cmd_investigate(a):
    from . import investigate
    investigate.run(a.work, a.complaint, render=not a.no_render, max_variants=a.max_variants, log=Log(a.work))


def cmd_analyze(a):
    os.makedirs(a.work, exist_ok=True)
    _analyze(a, Log(a.work))


def cmd_restore(a):
    log = Log(a.work)
    P, rep = _restore(a, log)
    if not a.no_verify:
        _autofix(a, _verify(a, P, rep, log), P, log)


def cmd_verify(a):
    log = Log(a.work)
    P = PL.load(os.path.join(a.work, "plan_used.json") if os.path.exists(os.path.join(a.work, "plan_used.json"))
                else os.path.join(a.work, "plan.json"))
    rep = json.load(open(os.path.join(a.work, "master.json"))) if os.path.exists(os.path.join(a.work, "master.json")) else {}
    _verify(a, P, rep, log)


def cmd_auto(a):
    os.makedirs(a.work, exist_ok=True)
    log = Log(a.work)
    _analyze(a, log)
    a.plan = None
    P, rep = _restore(a, log)
    _autofix(a, _verify(a, P, rep, log), P, log)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audiofix", description=f"live-audio restoration v{__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="which tools are available and what degrades without them")

    p = sub.add_parser("profile", help="build a reference profile (what good material sounds like)")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=["music", "speech"], default="music")
    p.add_argument("--files", nargs="*", help="reference recordings (studio tracks of the same artist/genre, clean speech)")
    p.add_argument("--itunes-artist", help="download official 30 s previews of this artist as references")
    p.add_argument("--song", action="store_true", help="the --files are the studio version of the very song being restored (analyze then finds the same passage and compares like with like)")
    p.add_argument("--itunes-song", help="with --itunes-artist: only this song (its studio version is the best target)")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--country", default="us")
    p.add_argument("--out", required=True, help="profile JSON path")
    p.add_argument("--work", help="scratch folder (separated references, downloads)")
    p.add_argument("--source", help="free-text note on where the references come from")
    p.add_argument("--no-stems", action="store_true", help="skip per-stem statistics")

    def common(q, need_input=True):
        if need_input:
            q.add_argument("input", help="audio or video file")
        q.add_argument("--work", required=True, help="work folder for this file")
        q.add_argument("--content", choices=["auto", "music", "speech"], default="auto",
                       help="force the content class instead of detecting songs vs MC")
        q.add_argument("--music-profile", default=None,
                       help="bundled profile name or a profile JSON path. Default: the studio version of this song, "
                            "looked up online from the tags (or --artist/--title); else the bundled jrock_band")
        q.add_argument("--artist", help="artist, for the studio-version lookup when the file has no tags")
        q.add_argument("--title", help="song title, for the studio-version lookup when the file has no tags")
        q.add_argument("--country", help="iTunes store to search first, e.g. jp, us (default: from the script)")
        q.add_argument("--no-lookup", action="store_true",
                       help="do not look the studio version up online; use --music-profile or the default")
        q.add_argument("--speech-profile", default="speech_ltass")
        q.add_argument("--no-stems", action="store_true", help="skip stem separation (mix-only fallbacks)")
        q.add_argument("--language", help="spoken language for Whisper, e.g. ja, en, zh (default: detect)")
        q.add_argument("--asr-model", help="faster-whisper model: tiny, base, small (default), medium, large-v3")
        q.add_argument("--codec-repair", choices=["auto", "on", "off"], default="auto",
                       help="Apollo codec repair: auto = when a low-bitrate codec wall (< 16 kHz) is found; slow on "
                            "CPU, about 6x real time")

    common(sub.add_parser("analyze", help="measure, detect content, write diagnosis.md and plan.json"))
    r = sub.add_parser("restore", help="render plan.json (or --plan) into restored.wav/m4a, then verify")
    r.add_argument("--work", required=True)
    r.add_argument("--plan")
    r.add_argument("--set", nargs="*", help="edits like music.delimit.depth=1.5 speech.crowd.enabled=false")
    r.add_argument("--no-verify", action="store_true")
    r.add_argument("--no-autofix", action="store_true", help="do not fix a WARN automatically (see review.md)")
    r.add_argument("--no-plots", action="store_true")
    v = sub.add_parser("verify", help="metrics, checks, A/B file and plots for a rendered work folder")
    v.add_argument("--work", required=True)
    v.add_argument("--no-plots", action="store_true")
    iv = sub.add_parser("investigate", help="a listener's complaint: hypotheses measured at every stage, one variant "
                                            "rendered per likely cause, and a listening file to choose from")
    iv.add_argument("--work", required=True)
    iv.add_argument("--complaint", required=True, help="the listener's words, in any language")
    iv.add_argument("--no-render", action="store_true", help="measure and rank only")
    iv.add_argument("--max-variants", type=int, default=3)
    au = sub.add_parser("auto", help="analyze + restore + verify in one go")
    common(au)
    au.add_argument("--set", nargs="*")
    au.add_argument("--no-autofix", action="store_true", help="do not fix a WARN automatically (see review.md)")
    au.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    {"check": cmd_check, "profile": cmd_profile, "analyze": cmd_analyze, "restore": cmd_restore,
     "verify": cmd_verify, "auto": cmd_auto, "investigate": cmd_investigate}[a.cmd](a)


if __name__ == "__main__":
    main()
