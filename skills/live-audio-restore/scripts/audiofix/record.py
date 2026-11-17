"""A run's record, and a step-by-step comparison of two runs.

Every analysis of another assistant's result used to be inferred from its finished audio: which steps ran, which
settings it changed and why could only be guessed. `bundle` packs the text a run leaves in its work folder (the
commands it ran, the diagnosis, the plan and every edit with its reason, the step ledger, the checks, the review),
the skill version and the machine's parts into one small zip, with no audio. `compare` lines two runs up, a bundle or
a work folder each, and names the first place they part: a different skill version, input or machine, a different
reference, a rule that decided differently, an edit, or the step in the ledger where the numbers split."""
import hashlib, io, json, os, platform, subprocess, sys, time, zipfile

from . import __version__

TEXT = ["commands.jsonl", "log.txt", "diagnosis.json", "diagnosis.md", "plan_default.json", "plan.json",
        "plan_used.json", "steps.json", "verification.json", "verification.md", "review.md", "master.json",
        "vocal_eq.json", "reference/lookup.json", "investigate/investigate.md", "investigate/variants.json",
        "previous/plan_used.json", "previous/verification.json", "previous/review.md",
        "autofix_rejected/plan_used.json", "autofix_rejected/verification.json"]
SKILL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def code_version():
    """The skill's version: __version__, a hash of its code, and the git commit when it is a checkout."""
    h = hashlib.sha256()
    pkg = os.path.dirname(os.path.abspath(__file__))
    for d in (pkg, os.path.dirname(pkg)):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    out = {"version": __version__, "code_sha": h.hexdigest()[:12]}
    try:
        g = subprocess.run(["git", "-C", SKILL, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           timeout=10)
        if g.returncode == 0:
            out["git"] = g.stdout.strip()
            dirty = subprocess.run(["git", "-C", SKILL, "status", "--porcelain", "--", "."], capture_output=True,
                                   text=True, timeout=10).stdout.strip()
            out["git_dirty"] = bool(dirty)
    except Exception:                                            # noqa: BLE001
        pass
    return out


def environment():
    pk = {}
    for m in ("numpy", "scipy", "soundfile", "numba", "torch", "demucs", "faster_whisper", "huggingface_hub"):
        try:
            mod = __import__(m)
            pk[m] = getattr(mod, "__version__", "?")
        except Exception:                                        # noqa: BLE001
            pk[m] = None
    ff = None
    try:
        ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10).stdout.split("\n")[0]
    except Exception:                                            # noqa: BLE001
        pass
    from . import analysis, device
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk, "ffmpeg": ff,
            "capabilities": analysis.capabilities(), "device": device.describe()}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def note_command(work, argv, started, exit_code=None):
    """commands.jsonl: one line when a command starts and one when it ends. A start with no end was cut off."""
    try:
        os.makedirs(work, exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)), "argv": argv}
        if exit_code is None:
            row["code"] = code_version()
        else:
            row.update(argv=argv[:1], done=True, exit=exit_code, seconds=round(time.time() - started, 1))
        with open(os.path.join(work, "commands.jsonl"), "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:                                            # noqa: BLE001 (a record must never stop a run)
        pass


def bundle(work, out=None, log=print):
    out = out or os.path.join(work, "run_record.zip")
    D = _json(os.path.join(work, "diagnosis.json")) or {}
    inp = {"name": os.path.basename(D.get("input", "")), "duration_s": D.get("duration_s"), "source": D.get("source")}
    if D.get("input") and os.path.exists(D["input"]):
        inp.update(size=os.path.getsize(D["input"]), sha256=_sha256(D["input"]))
    rec = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "work": os.path.abspath(work), "code": code_version(),
           "environment": environment(), "input": inp}
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("record.json", json.dumps(rec, ensure_ascii=False, indent=1))
        for f in TEXT:
            p = os.path.join(work, f)
            if os.path.isfile(p) and os.path.getsize(p) < 5 << 20:
                z.write(p, f)
                n += 1
    log(f"[bundle] {out}: {n} files and record.json, {os.path.getsize(out) // 1024} KB, no audio")
    log("[next] send this file to whoever compares runs (`audiofix.py compare THEIRS.zip MINE.zip`)")
    return out


def _json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def load_run(path):
    """A bundle (.zip) or a work folder -> {name: parsed json or text}."""
    r = {}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                raw = z.read(name).decode("utf-8", "replace")
                r[name] = _parse(name, raw)
    else:
        for f in TEXT:
            p = os.path.join(path, f)
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    r[f] = _parse(f, fh.read())
        src = (r.get("diagnosis.json") or {}).get("input")
        if src and os.path.exists(src):                     # a work folder on this machine: fingerprint its input
            r["record.json"] = {"input": {"sha256": _sha256(src)}}
    return r


def _parse(name, raw):
    if name.endswith(".jsonl"):
        return [json.loads(x) for x in raw.splitlines() if x.strip().startswith("{")]
    if name.endswith(".json"):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return raw


def _get(d, path):
    for k in path.split("."):
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list) and k.isdigit() and int(k) < len(d):
            d = d[int(k)]
        else:
            return None
    return d


def compare(pa, pb, names=("A", "B"), out=None, log=print):
    from .review import plan_diff
    A, B = load_run(pa), load_run(pb)
    na, nb = names
    L, first = [f"# Run comparison: {na} = {pa}, {nb} = {pb}", ""], []

    def row(label, a, b, key=None):
        unknown = a in (None, "?") or b in (None, "?")
        same = a == b
        L.append(f"| {label} | {a} | {b} | {'' if same else 'not recorded on one side' if unknown else 'DIFFERENT'} |")
        if not same and not unknown and key and key not in first:
            first.append(key)

    def code(R):
        c = (R.get("record.json") or {}).get("code")
        if not c:
            starts = [x for x in R.get("commands.jsonl") or [] if x.get("code")]
            c = starts[-1]["code"] if starts else {}
        return c or {}

    L += ["## Skill, machine, input", "", f"| | {na} | {nb} | |", "|---|---|---|---|"]
    ca, cb = code(A), code(B)
    row("skill code", ca.get("code_sha"), cb.get("code_sha"), "skill version")
    row("git commit", ca.get("git"), cb.get("git"))
    Da, Db = A.get("diagnosis.json") or {}, B.get("diagnosis.json") or {}
    row("parts installed", Da.get("capabilities"), Db.get("capabilities"), "machine")
    row("degraded (--degraded)", Da.get("degraded") or "no", Db.get("degraded") or "no", "machine")
    ia, ib = (A.get("record.json") or {}).get("input") or {}, (B.get("record.json") or {}).get("input") or {}
    if ia.get("sha256") or ib.get("sha256"):
        row("input sha256", (ia.get("sha256") or "?")[:16], (ib.get("sha256") or "?")[:16], "input file")
    row("input", f"{Da.get('duration_s')} s {_get(Da, 'source.codec')} {_get(Da, 'source.bit_rate')}",
        f"{Db.get('duration_s')} s {_get(Db, 'source.codec')} {_get(Db, 'source.bit_rate')}", "input")

    L += ["", "## Commands run", ""]
    for n, R in ((na, A), (nb, B)):
        cm = R.get("commands.jsonl") or []
        starts = [x for x in cm if not x.get("done")]
        ends = [x for x in cm if x.get("done")]
        L.append(f"- **{n}**: " + ("; ".join("`" + " ".join(x["argv"]) + "`" for x in starts) or "no record")
                 + (f" ({len(starts) - len(ends)} cut off before the end)" if len(starts) > len(ends) else ""))

    L += ["", "## What the analysis found", "", f"| | {na} | {nb} | |", "|---|---|---|---|"]
    for label, path, key in (("content", "processing_segments", "content"),
                             ("reference used", "reference.used", "reference"),
                             ("reference why", "reference.why", None),
                             ("reference passage similarity", "reference_section.similarity", None),
                             ("codec repair applied", "codec_repair.applied", "codec repair"),
                             ("cut-off Hz", "bandwidth.cutoff_hz", None),
                             ("whistle lines", "tonal_lines", None),
                             ("band replication", "sbr_copies", None)):
        va, vb = _get(Da, path), _get(Db, path)
        if path == "tonal_lines":
            va, vb = len(va or []), len(vb or [])
        if path == "sbr_copies":
            va = [s.get("shift_hz") for s in va or []]
            vb = [s.get("shift_hz") for s in vb or []]
        row(label, va, vb, key)

    L += ["", "## Plan", ""]
    pda, pdb = A.get("plan_default.json") or A.get("plan.json"), B.get("plan_default.json") or B.get("plan.json")
    if pda and pdb:
        d = plan_diff(pda, pdb)
        L.append("What the rules decided (default plans): " + ("the same" if not d else "; ".join(
            f"`{k}`: {a} vs {b}" for k, (a, b) in d.items())))
        if d:
            first.append("rules decided differently (different input, machine or skill version)")
    for n, R, pd in ((na, A, pda), (nb, B, pdb)):
        pu = R.get("plan_used.json")
        if pu and pd:
            d = plan_diff(pd, pu)
            why = pu.get("edit_reasons") or {}
            L.append(f"\nEdits by **{n}**: " + ("none" if not d else "; ".join(
                f"`{k}`: {a} -> {b} ({why.get(k, 'no reason recorded')})" for k, (a, b) in d.items())))
            if d:
                first.append(f"edits by {n}")
    pua, pub = A.get("plan_used.json"), B.get("plan_used.json")
    if pua and pub:
        d = plan_diff(pua, pub)
        L.append("\nPlans as rendered: " + ("the same" if not d else "; ".join(
            f"`{k}`: {a} vs {b}" for k, (a, b) in d.items())))

    L += ["", "## Step ledger (after each step)", ""]
    la, lb = _get(A.get("steps.json"), "ledger") or [], _get(B.get("steps.json"), "ledger") or []
    keys = [("lufs", 0.5), ("crest3s_db", 0.3), ("lra_lu", 0.3), ("mono_loss_db", 0.2), ("tone_rms_db", 0.5)]
    if la or lb:
        L += [f"| step | " + " | ".join(f"{k} {na} / {nb}" for k, _ in keys) + " |", "|---|" + "---|" * len(keys)]
        steps_b = {r["step"]: r for r in lb}
        split = None
        for r in la:
            o = steps_b.get(r["step"])
            cells = []
            for k, tol in keys:
                a, b = r.get(k), (o or {}).get(k)
                off = a is not None and b is not None and abs(a - b) > tol
                cells.append(f"{a} / {b}" + (" **" if off else ""))
                if off and split is None:
                    split = (r["step"], k, a, b)
            L.append(f"| {r['step']} | " + " | ".join(cells) + " |")
        only_b = [s for s in steps_b if s not in {r['step'] for r in la}]
        if only_b:
            L.append(f"\nSteps only in {nb}: {', '.join(only_b)}")
            first.append(f"steps only in {nb}: {', '.join(only_b)}")
        only_a = [r["step"] for r in la if r["step"] not in steps_b]
        if only_a:
            L.append(f"\nSteps only in {na}: {', '.join(only_a)}")
            first.append(f"steps only in {na}: {', '.join(only_a)}")
        if split:
            L.append(f"\nThe numbers first part after **{split[0]}**: {split[1]} {split[2]} vs {split[3]}")
            first.append(f"ledger splits after {split[0]} ({split[1]})")

    L += ["", "## Checks and result", "", f"| | {na} | {nb} | |", "|---|---|---|---|"]
    Va, Vb = A.get("verification.json") or {}, B.get("verification.json") or {}
    cb_ = {c["check"]: c for c in Vb.get("checks", [])}
    for c in Va.get("checks", []):
        o = cb_.get(c["check"], {})
        row(c["check"], ("PASS " if c["pass"] else "WARN ") + c.get("detail", ""),
            (("PASS " if o.get("pass") else "WARN ") + o.get("detail", "")) if o else "-")
    for k in ("lufs", "lra_lu", "crest3s_db", "plr_db", "true_peak_dbtp"):
        row(f"output {k}", _get(Va, f"output.loudness.{k}"), _get(Vb, f"output.loudness.{k}"))

    ra, rb = A.get("review.md") or "", B.get("review.md") or ""
    import re
    ka = set(re.findall(r"\*\*(\w+)\*\*:", ra))
    kb = set(re.findall(r"\*\*(\w+)\*\*:", rb))
    if ka != kb:
        L += ["", f"Review items only in {na}: {sorted(ka - kb) or '-'}; only in {nb}: {sorted(kb - ka) or '-'}"]

    L[1:1] = ["", "**Where they part, in order:** " + ("; ".join(first) if first else "nowhere measurable: same "
                                                       "skill, input, plan and ledger"), ""]
    text = "\n".join(L) + "\n"
    if out:
        with open(out, "w") as fh:
            fh.write(text)
        log(f"[compare] wrote {out}")
    log(L[2])
    return text
