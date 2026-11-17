"""Command line: check | profile | analyze | restore | verify | auto | investigate | bundle | compare.
Run `python3 scripts/audiofix.py -h`."""
import argparse, json, os, sys, time

from . import (__version__, analysis, audio_io, delimit, pipeline, plan as PL, profiles, ready, reference, report,
               review, stems, verify)


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
    rep = ready.check(cap)                                   # loads every model, downloading it once if needed
    miss = [k for k, v in rep.items() if not v["ok"]]
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
    for k, v in rep.items():
        print(f"{'READY' if v['ok'] else 'NOT READY'} {k}: {v['detail']}")
    if miss:
        print(ready.stop_text(miss, rep))
        sys.exit(3)
    print("[ok] every part of the full pipeline is installed and its model is on disk")


def _require(a, log):
    """Stop unless every part of the full pipeline is ready, or the user agreed to less (--degraded)."""
    rep = ready.check()
    miss = [k for k, v in rep.items() if not v["ok"]]
    if getattr(a, "no_stems", False) and "stems" not in miss:
        miss.append("stems")
        rep["stems"] = {"ok": False, "detail": "skipped with --no-stems"}
    if not miss:
        return None
    if not getattr(a, "degraded", None):
        log(ready.stop_text(miss, rep))
        sys.exit(3)
    log(f"[degraded] running without {', '.join(miss)}, as agreed: {a.degraded}")
    if "stems" in miss:
        a.no_stems = True
    if "codec_repair" in miss:
        a.codec_repair = "off"
    if "asr" in miss:
        os.environ["AUDIOFIX_NO_ASR"] = "1"
    return {"missing": miss, "why": a.degraded, "detail": {k: rep[k]["detail"] for k in miss}}


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
    degraded = _require(a, log)
    generic_ok = (getattr(a, "generic_reference", None) or "").strip() or None
    try:
        D = analysis.run(a.input, a.work, a.content, a.music_profile, a.speech_profile, not a.no_stems, log,
                         codec_repair=getattr(a, "codec_repair", "auto"),
                         lookup={"artist": a.artist, "title": a.title, "country": a.country,
                                 "lookup": not a.no_lookup}, degraded=degraded, generic_ok=generic_ok)
    except reference.NoStudioReference as e:
        _stop_reference(str(e), log)
    ref = D.get("reference") or {}
    if D.get("music") and ref.get("used") == reference.DEFAULT and not generic_ok:
        _stop_reference(ref.get("why", ""), log)
    P = PL.build(D)
    PL.save(P, os.path.join(a.work, "plan.json"))
    PL.save(P, os.path.join(a.work, "plan_default.json"))   # what the rules decided; every difference needs a reason
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


def _stop_reference(why, log):
    """A song is not restored against the generic profile unless the listener accepted it."""
    log(f"[stop] this is a song, and its tone target would be the bundled generic profile: {why}.")
    log("[stop] The studio version of this very song is the target the skill is built around: an assistant that "
        "went on with the generic profile delivered a different tone from one that used the studio version, on the "
        "same file. Do one of these, then run the same command again (the slow steps are cached):")
    log("[stop]   1. Ask the user for the artist and the title, and pass --artist \"...\" --title \"...\".")
    log("[stop]   2. No internet, or the song is not on iTunes: ask the user for the studio track (any copy, or 30 s of "
        "it), run `profile --name S --files STUDIO --song --out S.json`, then pass --music-profile S.json.")
    log("[stop]   3. Only if the user says there is no studio version (an arrangement, a cover, unreleased) or agrees "
        "to the generic target: pass --generic-reference \"<what the user said>\".")
    sys.exit(4)


def _reference_hint(P):
    """A [next] line when a song was matched to the generic profile: its own studio version is the better target."""
    ref = P.get("reference") or {}
    if not P.get("music") or ref.get("used") != reference.DEFAULT:
        return None
    return ("[next] the tone target is the bundled generic profile (" + ref.get("why", "") + "; accepted: "
            + str(ref.get("generic_ok")) + "). Say so in the delivery: the tone follows a genre average, not this "
            "song's studio version")


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


def _default_plan(work):
    return review.load_json(os.path.join(work, "plan_default.json")) or review.load_json(os.path.join(work, "plan.json"))


def _reasons(a, P, log):
    """Every setting that differs from what the rules decided needs its reason (--why). Hand-tuned plans made results
    depend on the assistant, and an edit nobody can explain cannot be judged or undone."""
    why = (getattr(a, "why", None) or "").strip()
    if a.set and not why:
        log("[stop] --set needs --why \"<the reason>\": the measurement, the WARN or the listener's words that call for "
            "this edit. Every edit to the default plan is listed with its reason in review.md and the delivery")
        sys.exit(2)
    reasons = dict(P.get("edit_reasons") or {})
    for s in a.set or []:
        reasons[s.split("=", 1)[0]] = why
    base = _default_plan(a.work)
    diff = review.plan_diff(base, P) if base else {}
    if why:
        for k in diff:
            reasons.setdefault(k, why)
    reasons = {k: v for k, v in reasons.items() if k in diff}
    unexplained = [k for k in diff if not reasons.get(k)]
    if unexplained:
        log(f"[stop] these settings differ from the default plan (plan_default.json) with no reason given: "
            f"{', '.join(unexplained)}. Run again with --why \"<the reason>\", or put them back")
        sys.exit(2)
    P["edit_reasons"] = reasons
    for k, r in reasons.items():
        log(f"[edit] {k} = {json.dumps(_get(P, k))}: {r}")
    return P


def _get(P, key):
    node = P
    for k in key.split("."):
        node = node.get(k) if isinstance(node, dict) else None
    return node


def _restore(a, log):
    P = PL.load(a.plan or os.path.join(a.work, "plan.json"))
    P = PL.override(P, a.set)
    P = _reasons(a, P, log)
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
                                               _default_plan(w), prev if prev["plan"] else None)
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
    sets, whys = [], []
    for fx, why in fixes:
        sets += [f"{k}={json.dumps(v)}" for k, v in fx.items()]
        whys.append(why)
        log(f"[autofix] {why}: " + ", ".join(f"{k}={v}" for k, v in fx.items()) + "; re-rendering once")
    b = argparse.Namespace(**vars(a))
    b.plan, b.set, b.why = os.path.join(a.work, "plan_used.json"), sets, "autofix: " + "; ".join(whys)
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
        q.add_argument("--no-stems", action="store_true",
                       help="skip stem separation (mix-only fallbacks); allowed only with --degraded")
        q.add_argument("--generic-reference", metavar="WHY",
                       help="restore a song against the bundled generic profile: only when the user said there is "
                            "no studio version, or agreed (otherwise a song with no studio reference stops)")
        q.add_argument("--degraded", metavar="WHY",
                       help="run without the parts check reports missing, after the user agreed: what failed to "
                            "install and that the user accepts a reduced result (shown in review.md)")
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
    r.add_argument("--why", help="the reason for the edits (required with --set, and for hand edits of plan.json): "
                                 "the measurement, WARN or listener's words that call for them")
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
    au.add_argument("--why", help="the reason for the --set edits (required with --set)")
    au.add_argument("--no-autofix", action="store_true", help="do not fix a WARN automatically (see review.md)")
    au.add_argument("--no-plots", action="store_true")
    bu = sub.add_parser("bundle", help="pack a work folder's record (commands, diagnosis, plans and edit reasons, "
                                       "step ledger, checks, review, skill version, machine) into a small zip, no audio")
    bu.add_argument("--work", required=True)
    bu.add_argument("--out", help="zip path (default WORK/run_record.zip)")
    co = sub.add_parser("compare", help="two runs (bundles or work folders) side by side, and where they first part")
    co.add_argument("a")
    co.add_argument("b")
    co.add_argument("--names", default="A,B", help="labels, e.g. GPT,Claude")
    co.add_argument("--out", help="write the comparison as Markdown here")
    a = ap.parse_args(argv)
    cmds = {"check": cmd_check, "profile": cmd_profile, "analyze": cmd_analyze, "restore": cmd_restore,
            "verify": cmd_verify, "auto": cmd_auto, "investigate": cmd_investigate, "bundle": cmd_bundle,
            "compare": cmd_compare}
    work = getattr(a, "work", None) if a.cmd not in ("bundle", "compare", "profile") else None
    t0, code = time.time(), 1
    if work:
        from . import record
        record.note_command(work, sys.argv[1:] if argv is None else list(argv), t0)
    try:
        cmds[a.cmd](a)
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        raise
    finally:
        if work:
            record.note_command(work, [a.cmd], t0, exit_code=code)


def cmd_bundle(a):
    from . import record
    record.bundle(a.work, a.out)


def cmd_compare(a):
    from . import record
    names = (a.names.split(",") + ["B"])[:2]
    record.compare(a.a, a.b, names=tuple(names), out=a.out)


if __name__ == "__main__":
    main()
