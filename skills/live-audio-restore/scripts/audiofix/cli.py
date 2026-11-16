"""Command line: check | profile | analyze | restore | verify | auto. Run `python3 scripts/audiofix.py -h`."""
import argparse, json, os, sys

from . import __version__, analysis, audio_io, delimit, pipeline, plan as PL, profiles, report, stems, verify


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
        prof = profiles.build(files, a.name, a.kind, work, with_stems=not a.no_stems, source=src or f"{len(files)} files")
    profiles.save(prof, a.out)
    print(f"profile '{a.name}' from {prof['count']} references -> {a.out}")


def _analyze(a, log):
    if getattr(a, "language", None):
        os.environ["AUDIOFIX_ASR_LANG"] = a.language
    if getattr(a, "asr_model", None):
        os.environ["AUDIOFIX_ASR_MODEL"] = a.asr_model
    D = analysis.run(a.input, a.work, a.content, a.music_profile, a.speech_profile, not a.no_stems, log,
                     codec_repair=getattr(a, "codec_repair", "auto"))
    P = PL.build(D)
    PL.save(P, os.path.join(a.work, "plan.json"))
    with open(os.path.join(a.work, "diagnosis.md"), "w") as fh:
        fh.write(report.diagnosis_md(D, P))
    try:
        report.overview_png(audio_io.load(os.path.join(a.work, "input.wav")), D, os.path.join(a.work, "overview.png"))
    except Exception as e:                                   # noqa: BLE001
        log(f"[analyze] overview plot skipped: {e}")
    log(f"[analyze] wrote diagnosis.md, plan.json, overview.png in {a.work}")
    return D, P


def _restore(a, log):
    P = PL.load(a.plan or os.path.join(a.work, "plan.json"))
    P = PL.override(P, a.set)
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
    log("[next] " + (f"{len(warns)} WARN: {'; '.join(warns)}. Before delivering, find the cause and act (SKILL.md "
                     f"step 6), then re-render" if warns else "all checks PASS"))
    log(f"[next] deliver {os.path.join(a.work, 'restored.wav')}, restored.m4a and AB.wav (the listener starts with "
        f"AB.wav); quote before/after numbers from verification.md and every plan edit you made, with its reason")
    return V


def cmd_analyze(a):
    os.makedirs(a.work, exist_ok=True)
    _analyze(a, Log(a.work))


def cmd_restore(a):
    log = Log(a.work)
    P, rep = _restore(a, log)
    if not a.no_verify:
        _verify(a, P, rep, log)


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
    _verify(a, P, rep, log)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audiofix", description=f"live-audio restoration v{__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="which tools are available and what degrades without them")

    p = sub.add_parser("profile", help="build a reference profile (what good material sounds like)")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=["music", "speech"], default="music")
    p.add_argument("--files", nargs="*", help="reference recordings (studio tracks of the same artist/genre, clean speech)")
    p.add_argument("--itunes-artist", help="download official 30 s previews of this artist as references")
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
        q.add_argument("--music-profile", default="jrock_band", help="bundled profile name or a profile JSON path")
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
    r.add_argument("--no-plots", action="store_true")
    v = sub.add_parser("verify", help="metrics, checks, A/B file and plots for a rendered work folder")
    v.add_argument("--work", required=True)
    v.add_argument("--no-plots", action="store_true")
    au = sub.add_parser("auto", help="analyze + restore + verify in one go")
    common(au)
    au.add_argument("--set", nargs="*")
    au.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    {"check": cmd_check, "profile": cmd_profile, "analyze": cmd_analyze, "restore": cmd_restore,
     "verify": cmd_verify, "auto": cmd_auto}[a.cmd](a)


if __name__ == "__main__":
    main()
