"""Human-readable reports: diagnosis.md (+ overview.png) and verification.md."""
import numpy as np

from . import SR


def _v(ok):
    return "OK" if ok else "PROBLEM"


def diagnosis_md(D, P):
    L = D["loudness"]; st = D["stereo"]; bw = D["bandwidth"]
    out = [f"# Diagnosis: {D['source'].get('path', D['input'])}", ""]
    s = D["source"]
    out += [f"- Source: {s.get('codec')} {s.get('profile') or ''}, {(s.get('bit_rate') or 0) // 1000} kb/s, "
            f"{s.get('sample_rate')} Hz, {s.get('channels')} ch, {D['duration_s']} s",
            f"- Tools available: {D['capabilities']}",
            f"- Stem separation residual: {D.get('separation_residual_db', 'n/a')} dB", ""]
    out += ["## Content map", "", "| from (s) | to (s) | detected | processed as |", "|---|---|---|---|"]
    proc = P["segments"]
    for a, b, c in D["segments"]:
        pc = next((p[2] for p in proc if p[0] <= (a + b) / 2 < p[1]), c)
        out.append(f"| {a:.1f} | {b:.1f} | {c} | {pc} |")
    if D["sections"]:
        out += ["", f"Music sections with their own tonal treatment (split at recording-chain changes): {D['sections']}"]
    out += ["", "## Global findings", "", "| check | measured | rule | verdict |", "|---|---|---|---|",
            f"| loudness | {L['lufs']} LUFS, LRA {L['lra_lu']} LU | LRA < 4 LU or PLR < 11 dB = limited | "
            f"{'limited' if (L['lra_lu'] < 4 or L['plr_db'] < 11) else 'open'} |",
            f"| peak-to-loudness | {L['plr_db']} dB (true peak {L['true_peak_dbtp']} dBTP) | | |",
            f"| clipping | {D['clipping']['flat_runs_per_min']} flat runs/min | >= 20/min | {_v(D['clipping']['flat_runs_per_min'] < 20)} |",
            f"| mains hum | {D['hum']['mains_hz']} Hz, excess {D['hum']['excess_db']} dB | fund >= 10 or 2 harm >= 8 dB | {_v(not D['hum']['present'])} |",
            f"| bandwidth | {bw['cutoff_hz']} Hz, edge {bw['edge_drop_db']} dB | brick wall below 18.5 kHz | {_v(not (bw['brick_wall'] and bw['cutoff_hz'] < 18500))} |",
            f"| codec notches | {D['codec_notches'] or 'none'} | | |",
            f"| stereo | {st['kind']}, side-mid {st['side_minus_mid_db']} dB | | |"]
    if st["kind"] != "mono":
        out += [f"| L-R correlation by band | {st.get('band_corr')} | | |",
                f"| mid-band anti-phase | worst {st.get('mid_corr_min')}, negative in {st.get('mid_corr_neg_pct')} % of windows | >= 10 % | {_v(st.get('mid_corr_neg_pct', 0) < 10)} |",
                f"| mono compatibility | mono-sum loss {st.get('mono_loss_db')} dB | < -1.5 dB | {_v(st.get('mono_loss_db', 0) >= -1.5)} |"]
    out += [f"| noise floor | quietest 5 %: {D['noise']['p5_dbfs']} dBFS, median {D['noise']['p50_dbfs']} dBFS | | |"]
    mu = D.get("music")
    if mu:
        out += ["", "## Music", "", f"- {mu['duration_s']} s; loudness {mu['loudness']}",
                f"- Tone vs profile `{D['profiles']['music']}`: RMS deviation {mu['tonal']['rms_db']} dB; "
                f"by band (dB, + means too much): {mu['tonal']['by_band']}"]
        if mu.get("stems"):
            out += ["", "| stem | level re mix | anti-phase energy | worst corr | median corr | anti-phase by band (cos) |",
                    "|---|---|---|---|---|---|"]
            for k, v in mu["stems"].items():
                s2 = v["stereo"]
                out.append(f"| {k} | {v['level_re_mix_db']} dB | {s2.get('anti_pct', 0)} % | {s2.get('corr_min')} | "
                           f"{s2.get('corr_median')} | {s2.get('anti_band_cos', '')} |")
            out += ["", f"- Who carries the 3-10 kHz energy (harshness): {mu.get('treble_share_pct')}"]
    sp = D.get("speech")
    if sp:
        out += ["", "## Speech (MC)", "", f"- {sp['duration_s']} s; loudness {sp['loudness']}",
                f"- {sp['metrics']}",
                f"- Voice vs speech spectrum: RMS deviation {sp['tonal']['rms_db']} dB; by band {sp['tonal']['by_band']}"]
    out += ["", "## Plan (edit plan.json, or pass --set key=value to restore)", ""]
    for part in ("repair", "music", "speech"):
        if not P.get(part):
            continue
        out += [f"### {part}", ""]
        for k, v in P[part].items():
            if isinstance(v, dict):
                on = v.get("enabled", True if "stems" not in v else bool(v["stems"]))
                params = {kk: vv for kk, vv in v.items() if kk not in ("why", "enabled")}
                out.append(f"- **{k}**: {'ON' if on else 'off'} {params if params else ''} -- {v.get('why', '')}")
            else:
                out.append(f"- **{k}**: {v}")
        out.append("")
    out += ["### master", "", f"- {P['master']['why']}", ""]
    if P.get("notes"):
        out += ["## Notes", ""] + [f"- {n}" for n in P["notes"]]
    return "\n".join(out) + "\n"


def overview_png(x, D, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    from .measure import rms_db
    m = x.mean(axis=1)
    step = max(1, len(m) // (SR * 600))                    # keep the plot light on long files
    f, t, Z = stft(m[::step], SR / step, nperseg=4096 // step if step < 4 else 1024, noverlap=None)
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw=dict(height_ratios=[3, 1, 1]), sharex=True)
    ax[0].pcolormesh(t, f / 1000, 20 * np.log10(np.abs(Z) + 1e-9), vmin=-110, vmax=-20, cmap="magma", shading="auto")
    ax[0].set_ylabel("kHz"); ax[0].set_title("input: spectrogram, level, stereo width; shaded = content class")
    lv, tt = rms_db(x, 0.4, 0.1)
    ax[1].plot(tt, lv, lw=0.6); ax[1].set_ylabel("RMS dBFS"); ax[1].grid(alpha=.3)
    s, mm = (x[:, 0] - x[:, 1]) / 2, (x[:, 0] + x[:, 1]) / 2
    w = int(SR); r = []
    for i in range(0, len(x) - w, w // 2):
        r.append(20 * np.log10((np.std(s[i:i + w]) + 1e-9) / (np.std(mm[i:i + w]) + 1e-9)))
    ax[2].plot(np.arange(len(r)) * 0.5, r, lw=0.6, color="C2"); ax[2].set_ylabel("side-mid dB"); ax[2].grid(alpha=.3)
    colors = {"music": "#3a7", "speech": "#37c", "gap": "#999"}
    for a, b, c in D["segments"]:
        for k in (1, 2):
            ax[k].axvspan(a, b, color=colors[c], alpha=0.12)
    for a, b in D.get("sections", []):
        for k in (1, 2):
            ax[k].axvline(a, color="k", lw=0.6, ls="--")
    ax[2].set_xlabel("s")
    plt.tight_layout(); plt.savefig(path, dpi=60); plt.close(fig)


def verification_md(V, P, rep):
    out = ["# Verification", "", "| check | result | detail |", "|---|---|---|"]
    for c in V["checks"]:
        out.append(f"| {c['check']} | {'PASS' if c['pass'] else 'WARN'} | {c['detail']} |")
    out += ["", "| | input | restored |", "|---|---|---|"]
    for k in ("lufs", "true_peak_dbtp", "plr_db", "lra_lu", "crest3s_db"):
        out.append(f"| {k} | {V['input']['loudness'][k]} | {V['output']['loudness'][k]} |")
    si, so = V["input"]["stereo"], V["output"]["stereo"]
    for k in ("side_minus_mid_db", "mid_corr_min", "mid_corr_mean", "mono_loss_db"):
        if k in si:
            out.append(f"| {k} | {si.get(k)} | {so.get(k)} |")
    out.append(f"| bandwidth (Hz) | {V['input']['bandwidth']['cutoff_hz']} | {V['output']['bandwidth']['cutoff_hz']} |")
    if V.get("music"):
        mi, mo = V["music"]["input"]["tonal"], V["music"]["output"]["tonal"]
        out += ["", "## Music: deviation from the reference profile, dB (+ = too much)", "",
                "| band (Hz) | input | restored |", "|---|---|---|"]
        for b in mi["by_band"]:
            out.append(f"| {b} | {mi['by_band'][b]:+.1f} | {mo['by_band'].get(b, float('nan')):+.1f} |")
        out.append(f"| RMS | {mi['rms_db']} | {mo['rms_db']} |")
    if V.get("speech"):
        si, so = V["speech"]["input"], V["speech"]["output"]
        out += ["", "## Speech", "", f"- input: {si['metrics']}, tone RMS dev {si['tonal']['rms_db']} dB",
                f"- restored: {so['metrics']}, tone RMS dev {so['tonal']['rms_db']} dB"]
    out += ["", "## Master", "", f"- {rep}"]
    if V.get("ab_times_s"):
        out += ["", "## A/B file", "", "AB.wav: for each passage the input, then the restored version, loudness-matched "
                "per pair. Passages start at (s): " + ", ".join(f"{t:.0f}" for t in V["ab_times_s"])]
    return "\n".join(out) + "\n"
