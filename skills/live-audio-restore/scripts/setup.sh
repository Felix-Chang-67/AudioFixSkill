#!/usr/bin/env bash
# Installs what the full pipeline uses; the skill requires all of it. Safe to re-run.
# Usage: bash scripts/setup.sh  (PYTHON=python3.x to pick one). Ends with `check`, which exits 3 if a part is missing.
set -euo pipefail
PY=${PYTHON:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== core (always needed)"
$PY -m pip install --quiet numpy scipy soundfile matplotlib numba

echo "== stem separation (demucs + torch, CPU is fine)"
$PY -m pip install --quiet demucs torch torchaudio || echo "!! demucs/torch failed: required; analyze will refuse to run until it is installed"

echo "== neural de-limiter (De-limiter, WASPAA 2023)"
$PY -m pip install --quiet --no-deps asteroid==0.7.0 || true
$PY -m pip install --quiet asteroid-filterbanks einops dotmap || true
DL="${AUDIOFIX_DELIMITER:-$HOME/.cache/audiofix/De-limiter}"
if [ ! -f "$DL/weight/all.pth" ]; then
  mkdir -p "$(dirname "$DL")"
  git clone --depth 1 https://github.com/jeonchangbin49/De-limiter "$DL" || echo "!! could not clone De-limiter: required; fix and re-run"
fi

echo "== codec repair for low-bitrate rips (Apollo, CC BY-SA 4.0; its checkpoint downloads on first use)"
$PY -m pip install --quiet huggingface_hub || true
AP="${AUDIOFIX_APOLLO:-$HOME/.cache/audiofix/Apollo}"
if [ ! -f "$AP/look2hear/models/apollo.py" ]; then
  mkdir -p "$(dirname "$AP")"
  git clone --depth 1 https://github.com/JusperLee/Apollo "$AP" || echo "!! could not clone Apollo: required; fix and re-run"
fi

echo "== speech recognition for MC talker timing and the intelligibility check (faster-whisper)"
$PY -m pip install --quiet faster-whisper || echo "!! faster-whisper failed: required; fix and re-run"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "!! ffmpeg not found: install it (apt install ffmpeg | brew install ffmpeg | winget install ffmpeg)"
fi
echo "== every part ready, and its model downloaded (demucs, Apollo, Whisper), so later runs work offline"
$PY "$HERE/audiofix.py" check
