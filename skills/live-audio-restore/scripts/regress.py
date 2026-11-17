#!/usr/bin/env python3
"""Regression run: `auto` on every recording of evals/regression.json that can be found, then each expectation
checked against what the run wrote. Run it after changing a rule: a fix for one song must not break another.

    python3 scripts/regress.py                         # the two synthetic clips that ship with the skill
    python3 scripts/regress.py --media ~/my_recordings # plus your copies of the other recordings
    python3 scripts/regress.py --only tenohira_acoustic --check-only   # re-check an existing work folder

Work folders (default ./regress_work/<case>) are kept: a second run reuses codec repair, stems and the reference.
Exit code 1 when an expectation fails."""
import argparse, fnmatch, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)


def find(case, media):
    if case.get("file"):
        p = os.path.join(SKILL, case["file"])
        return p if os.path.exists(p) else None
    if not media or not os.path.isdir(media):
        return None
    for name in sorted(os.listdir(media)):
        if any(fnmatch.fnmatch(name, pat) for pat in case.get("match", [])):
            return os.path.join(media, name)
    return None


def load(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def context(work):
    V = load(os.path.join(work, "verification.json"))
    V["warns"] = sum(not c["pass"] for c in V.get("checks", [])) if V.get("checks") else None
    kinds = []
    if os.path.exists(os.path.join(work, "review.md")):
        kinds = re.findall(r"^\d+\. \*\*([a-z]+)\*\*", open(os.path.join(work, "review.md")).read(), re.M)
    return {"diagnosis": load(os.path.join(work, "diagnosis.json")), "plan": load(os.path.join(work, "plan.json")),
            "plan_used": load(os.path.join(work, "plan_used.json")), "verify": V,
            "steps": load(os.path.join(work, "steps.json")), "review": {"kinds": kinds}}


def resolve(ctx, path):
    """Values at a dotted path; `*` expands a dict's values or a list's items; numbers index lists."""
    vals = [ctx]
    for key in path.split("."):
        nxt = []
        for v in vals:
            if key == "*":
                nxt += list(v.values()) if isinstance(v, dict) else list(v) if isinstance(v, list) else []
            elif isinstance(v, dict) and key in v:
                nxt.append(v[key])
            elif isinstance(v, list) and key.isdigit() and int(key) < len(v):
                nxt.append(v[int(key)])
        vals = nxt
    return vals


def check(ctx, e):
    vals = resolve(ctx, e["path"])
    if e.get("info"):
        return None, f"{vals[0] if len(vals) == 1 else vals}"
    if not vals or any(v is None for v in vals):
        return False, "missing"
    v = vals if "*" in e["path"] else vals[0]
    if "eq" in e:
        return v == e["eq"], f"{v}"
    if "contains" in e:
        return e["contains"] in v, f"{v}"
    if "near" in e:                                   # a list holds a number within tol of each value
        miss = [n for n in e["near"] if not any(abs(x - n) <= e.get("tol", 1.0) for x in v)]
        return not miss, f"missing {miss}" if miss else f"{len(v)} values, all {len(e['near'])} found"
    if "endswith" in e:
        return str(v).endswith(e["endswith"]), f"{v}"
    if "approx" in e:
        return abs(v - e["approx"]) <= e["tol"], f"{v}"
    if "abs_max" in e:
        return all(abs(x) <= e["abs_max"] for x in vals), f"{[round(x, 1) for x in vals]}"
    ok = True
    if "min" in e:
        ok &= all(x >= e["min"] for x in vals)
    if "max" in e:
        ok &= all(x <= e["max"] for x in vals)
    return ok, f"{v}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media", help="folder with your copies of the recordings named in regression.json")
    ap.add_argument("--work", default="regress_work", help="work folders, one per case (kept, caches reused)")
    ap.add_argument("--only", help="comma-separated case names")
    ap.add_argument("--check-only", action="store_true", help="do not run, only check existing work folders")
    a = ap.parse_args()
    base = load(os.path.join(SKILL, "evals", "regression.json"))
    only = set(a.only.split(",")) if a.only else None
    failed, ran = 0, 0
    for case in base["cases"]:
        if only and case["name"] not in only:
            continue
        src = find(case, a.media)
        work = os.path.join(a.work, case["name"])
        if not src and not (a.check_only and os.path.isdir(work)):
            print(f"SKIP  {case['name']}: input not found")
            continue
        if not a.check_only:
            os.makedirs(work, exist_ok=True)
            cmd = [sys.executable, os.path.join(HERE, "audiofix.py"), "auto", src, "--work", work] + case.get("args", [])
            print(f"RUN   {case['name']}: {' '.join(cmd[2:])}", flush=True)
            with open(os.path.join(work, "regress.log"), "w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"FAIL  {case['name']}: audiofix exited with {rc} (see {work}/regress.log)")
                failed += 1
                continue
        ran += 1
        ctx = context(work)
        for e in case["expect"]:
            ok, got = check(ctx, e)
            tag = "INFO" if ok is None else "PASS" if ok else "FAIL"
            failed += ok is False
            print(f"{tag}  {case['name']}: {e['path']} -> {got}   ({e.get('why', '')})")
    print(f"\n{ran} case(s) checked, {failed} failure(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
