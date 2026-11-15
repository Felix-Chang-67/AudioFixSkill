# AGENTS.md

This repository contains one agent skill: `skills/live-audio-restore/`. It diagnoses and restores low-quality live
recordings: concert songs, spoken MC / stage talk, or both in one file.

## When the user wants a recording fixed

Use the skill whenever the user asks to fix, restore, clean up, remaster or improve a live recording or a talk clip,
or says it sounds bad, muddy, harsh, squashed, thin, phasey, too quiet or drowned by the crowd. The user may write in
any language (for example 修复 / 精修 / 音质 / 现场录音).

1. Read `skills/live-audio-restore/SKILL.md` and follow its workflow: analyze → plan → restore → verify.
2. Run commands from `skills/live-audio-restore/`:
   - First time: `bash scripts/setup.sh`, or at least `python3 scripts/audiofix.py check`.
   - Then run `python3 scripts/audiofix.py auto INPUT --work WORK`, or the steps one by one.
3. Read the right reference at the right time:
   - `references/diagnosis.md` before you interpret `diagnosis.md`;
   - `references/tuning.md` when the user gives listening feedback;
   - `references/other-ai.md` when there is no internet, no PyTorch or no ffmpeg (`--no-stems`, WAV input).
4. You cannot hear audio:
   - Report only changes that `verification.md` measured.
   - Hand over `restored.wav`, `restored.m4a` and `AB.wav`, and ask the user to listen to `AB.wav` first.
5. Trust the default plan:
   - Edit it only for a WARN, a measurement the rules missed, or the listener's words, and report every edit with
     its reason.
   - Hand-tuned plans made results depend on the assistant: 3 dB apart in loudness, 3 LU in dynamics.
   - Act on every WARN, and on the `[next]` lines `restore` prints, before delivering.
6. Never overwrite the user's source file; everything goes into the work folder.
7. Explain the result in the user's language, in plain words.

## Working on the code

- **Library:** `skills/live-audio-restore/scripts/audiofix/`, pure Python.
  - Core: numpy, scipy, soundfile.
  - Optional: ffmpeg, demucs + torch, De-limiter, faster-whisper, numba. Each one degrades gracefully.
- **Decision rules:** `plan.py`. Every threshold there is documented in `references/diagnosis.md`; change both
  together.
- **Test clips:** `python3 scripts/make_test_clip.py song|mc out.m4a` makes a synthetic clip with known problems.
  - Eval cases live in `skills/live-audio-restore/evals/`.
  - A change should keep every check in `verification.md` at PASS on both eval clips.
- **Environment switches:**
  - `AUDIOFIX_NO_NUMBA=1` and `AUDIOFIX_NO_ASR=1` test the fallbacks.
  - `AUDIOFIX_ASR_MODEL` and `AUDIOFIX_ASR_LANG` set the Whisper model and language.
  - `AUDIOFIX_DELIMITER` is the De-limiter checkout path.
