"""Is this machine ready for the full pipeline: every part installed, and its model on disk?

The skill used to run whatever this machine had and report the rest. An assistant in a sandbox without PyTorch
delivered a render with no stem separation and no codec repair (the vocal as wide as in the original, the codec's
9-13 kHz holes kept) and never said so; the same song restored with everything sounded clearly better. Those
assistants can install the parts when they are told to. So `analyze` and `auto` stop with exit code 3 and the
install command when a part is missing, and run a reduced pipeline only with `--degraded "<why>"`, after the
listener agreed.

A part counts as ready when its model loads now: downloaded if the network allows, read from the cache if not. So
`check` (run by setup.sh) also fetches every model once, and a later session without internet still runs."""
import os

REQUIRED = ("ffmpeg", "stems", "delimiter", "codec_repair", "asr")
HOW = {
    "ffmpeg": "install ffmpeg (apt install ffmpeg | brew install ffmpeg | winget install ffmpeg | conda install ffmpeg)",
    "stems": "bash scripts/setup.sh (pip install demucs torch torchaudio; the htdemucs model downloads once)",
    "delimiter": "bash scripts/setup.sh (clones De-limiter with its weights, installs asteroid)",
    "codec_repair": "bash scripts/setup.sh (clones Apollo, pip install huggingface_hub; the checkpoint downloads once)",
    "asr": "bash scripts/setup.sh (pip install faster-whisper; the Whisper model downloads once)",
}
LOSS = {
    "ffmpeg": "no m4a/mp3/video input, no AAC copy, no stationary denoise or declip",
    "stems": "no per-instrument work: no vocal width, stem balance, per-stem tone or dynamics (the largest loss)",
    "delimiter": "no neural de-limiting on limited sources: less punch comes back",
    "codec_repair": "no codec repair on low-bitrate rips: the highs keep a sparse, metallic texture",
    "asr": "no word-accurate talker timing for MCs, no check that the words are still recognised",
}


def _model(name):
    """(ok, detail): does this part's model load now?"""
    try:
        if name == "stems":
            from demucs.pretrained import get_model
            get_model("htdemucs")
        elif name == "codec_repair":
            from huggingface_hub import hf_hub_download
            from .codec import CKPT_FILE, CKPT_REPO
            try:
                hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE, local_files_only=True)
            except Exception:                                    # noqa: BLE001 (not cached yet: fetch it)
                hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE)
        elif name == "asr":
            from faster_whisper.utils import download_model
            m = os.environ.get("AUDIOFIX_ASR_MODEL", "small")
            try:
                download_model(m, local_files_only=True)
            except Exception:                                    # noqa: BLE001
                download_model(m)
        return True, "model ready"
    except Exception as e:                                       # noqa: BLE001
        return False, f"installed, but its model could not be loaded or downloaded ({str(e)[:160]})"


def check(cap=None):
    """{part: {"ok", "detail"}} for every required part. cap: analysis.capabilities(), computed if not given."""
    if cap is None:
        from .analysis import capabilities
        cap = capabilities()
    out = {}
    for k in REQUIRED:
        if not cap.get(k):
            out[k] = {"ok": False, "detail": "not installed"}
        elif k in ("stems", "codec_repair", "asr"):
            ok, det = _model(k)
            out[k] = {"ok": ok, "detail": det}
        else:
            out[k] = {"ok": True, "detail": "ready"}
    return out


def stop_text(missing, report):
    lines = [f"[stop] this machine is not ready for the full pipeline; missing: {', '.join(missing)}."]
    for k in missing:
        lines.append(f"[stop]   {k}: {report[k]['detail']}. Install: {HOW[k]}. Without it: {LOSS[k]}")
    lines.append("[stop] The skill requires the full install. Run `bash scripts/setup.sh` (internet needed once; it "
                 "also downloads the models), check that `python3 scripts/audiofix.py check` shows every part ready, "
                 "then run this command again. Do not work around it with --no-stems or by skipping steps.")
    lines.append("[stop] Only if the install fails and you cannot fix it: tell the user exactly what failed and what "
                 "the result will lack (the lines above), and only with their agreement run the same command with "
                 "--degraded \"<what failed and that the user agreed>\".")
    return "\n".join(lines)
