# AGENTS.md

This repository contains one agent skill: `skills/live-audio-restore/`. It diagnoses and restores low-quality live
recordings: concert songs, spoken MC / stage talk, or both in one file.

## When the user wants a recording fixed

Use the skill whenever the user asks to fix, restore, clean up, remaster or improve a live recording or a talk clip,
or says it sounds bad, muddy, harsh, squashed, thin, phasey, too quiet or drowned by the crowd. The user may write in
any language (for example 修复 / 精修 / 音质 / 现场录音).

1. Read `skills/live-audio-restore/SKILL.md` and follow its workflow: analyze → plan → restore → verify.
2. Run commands from `skills/live-audio-restore/`:
   - First time: `bash scripts/setup.sh`. It is required: it installs every part and downloads the models.
     `python3 scripts/audiofix.py check` must end with `[ok] every part of the full pipeline is installed`.
   - `analyze` and `auto` stop (exit code 3) when a part is missing. Install it; do not work around it with
     `--no-stems`, `--codec-repair off` or skipped steps.
   - If the install fails and you cannot fix it, tell the user what failed and what the result will lack. Only with
     their agreement run with `--degraded "<what failed; the user agreed>"`, and repeat what is missing in the
     delivery.
   - Codec repair takes about 6x the song's length on a CPU. If your tool stops long commands, run it in the
     background and follow `WORK/log.txt`; if it is cut off, run the same command again: it resumes.
   - Then run `python3 scripts/audiofix.py auto INPUT --work WORK --content music` for a song, or
     `--content speech` for an MC or talk, or the steps one by one.
   - Leave out `--content` only for a file that holds both. When the user says what the file is, that beats the
     automatic split.
   - For a song, `auto` looks up the studio version of that very song online and makes it the tone target. It reads
     the artist and title from the file's tags, or from `--artist` / `--title`, and fetches the official iTunes
     preview. It drops the preview if its passage is not found in the recording.
   - Without network, tags or a match, it falls back to the bundled profile and prints a `[next]` line. Then ask the
     user for the studio track (`references/other-ai.md`).
3. Read the right reference at the right time:
   - `references/diagnosis.md` before you interpret `diagnosis.md`;
   - `references/tuning.md` when the user gives listening feedback;
   - `references/other-ai.md` when the install fails (no internet, no PyTorch, no ffmpeg).
4. You cannot hear audio:
   - Report only changes that `verification.md` measured.
   - Hand over `restored.wav`, `restored.m4a` and `AB.wav`, and ask the user to listen to `AB.wav` first.
5. Trust the default plan:
   - Edit it only for a WARN, a measurement the rules missed, or the listener's words, and report every edit with
     its reason.
   - Every edit needs its reason on the command line: `restore --work WORK --set key=value --why "<reason>"`.
     `restore` refuses an edit without one, hand edits of `plan.json` included (they are compared with
     `plan_default.json`).
   - Hand-tuned plans made results depend on the assistant: 3 dB apart in loudness, 3 LU in dynamics.
   - Act on every WARN, and on the `[next]` lines `restore` prints, before delivering.
6. Work carefully. All checks PASS is not the end:
   - Read the `[review]` lines after `analyze` and after `restore` (also in `review.md`), and answer every one in the
     delivery. They cover applause, missing instruments, extreme peaks, a weak reference, rebuilt highs, steps at
     their limits, and your edits to the default plan.
   - `review.md` traces each WARN to the step that caused it (the ledger in `steps.json`) and names its knob.
   - When that step has a safe remedy, `restore` applies it once on its own (`[autofix]` lines) and keeps the result
     only if the WARNs go down. Name that edit and its reason in the delivery. `--no-autofix` turns this off.
   - Change one knob at a time and re-render. The last render stays in `previous/` for comparison.
   - `missing` items say what this machine could not do (no PyTorch, no internet, no ffmpeg). Tell the user what the
     result lacks because of it.
7. When the listener says something sounds wrong, investigate before changing anything:
   - Run `python3 scripts/audiofix.py investigate --work WORK --complaint "<their exact words>"`.
   - It measures every likely cause, ranks them, renders one variant per cause and writes `investigate/ABC.wav`.
   - Send that file, explain the evidence, ask which version is best, then apply that version's settings
     (`investigate/investigate.md`) with `--why "the listener chose version N"`. Do not pick a knob from the words
     alone.
8. Never overwrite the user's source file; everything goes into the work folder.
9. Explain the result in the user's language, in plain words.
10. After delivering, run `python3 scripts/audiofix.py bundle --work WORK`. `WORK/run_record.zip` holds the run's
    record, with no audio: every command, the plans and edit reasons, the step ledger, the checks, the review, the
    skill version and the machine. Send it when the user asks how the result was made or wants it compared with
    another assistant's; `compare A.zip B.zip` shows where two runs part.

## Working on the code

- **Library:** `skills/live-audio-restore/scripts/audiofix/`, pure Python.
  - Core: numpy, scipy, soundfile.
  - Required for a run (`ready.py`): ffmpeg, demucs + torch, De-limiter, Apollo, faster-whisper, each with its
    model on disk. Their fallbacks still exist, for `--degraded` runs only. numba is optional (speed).
- **Decision rules:** `plan.py`. Every threshold there is documented in `references/diagnosis.md`; change both
  together.
- **Test clips:** `python3 scripts/make_test_clip.py song|mc out.m4a` makes a synthetic clip with known problems.
  - Eval cases live in `skills/live-audio-restore/evals/`.
  - A change should keep every check in `verification.md` at PASS on both eval clips.
- **Regression baseline:** after changing a rule, run `python3 scripts/regress.py` from `skills/live-audio-restore/`.
  - It checks `evals/regression.json`: what correct runs concluded on every recording the skill was tuned on.
  - The two synthetic clips ship with the repository (a full install is needed). `--media DIR` adds your copies of
    the other recordings (not distributed).
  - Every expectation must stay PASS. A fix for one song must not break another.
- **Environment switches:**
  - `AUDIOFIX_NO_NUMBA=1` and `AUDIOFIX_NO_ASR=1` test the fallbacks (with `--degraded "test"` for the second).
  - `AUDIOFIX_ASR_MODEL` and `AUDIOFIX_ASR_LANG` set the Whisper model and language.
  - `AUDIOFIX_DELIMITER` is the De-limiter checkout path; `AUDIOFIX_APOLLO` is the Apollo checkout path.
  - `AUDIOFIX_DEVICE=auto|cuda|mps|cpu` picks the device for Apollo and demucs. Auto uses a CUDA or MPS GPU when
    there is one; a missing or failing GPU falls back to the CPU.
