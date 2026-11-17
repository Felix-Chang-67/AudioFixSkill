# Using this skill with other assistants, or without one

The skill has two parts:

- the instructions: `SKILL.md` and these references, in plain Markdown;
- a command-line tool: `scripts/audiofix.py`, in plain Python.

Nothing in it depends on one vendor. What changes between assistants is whether they can run commands, install
packages and reach the internet.

## Agents that run commands (Codex CLI, Claude Code, Gemini CLI, Cursor, Copilot agent mode, Aider, ...)

1. Put the repository in the workspace. `AGENTS.md` at the repository root points the agent to `SKILL.md`; many
   agents read `AGENTS.md` automatically.
2. Install once:
   ```bash
   bash skills/live-audio-restore/scripts/setup.sh
   ```
   This needs internet for pip, the De-limiter clone, and the demucs and Whisper models on first use.
3. Ask in plain words: "restore this live recording", "the MC is drowned by the crowd", and so on. The agent follows
   the workflow in `SKILL.md`.

Tools that support Agent Skills folders (a `SKILL.md` with `name`/`description` frontmatter, plus `scripts/` and
`references/`) can load `skills/live-audio-restore/` directly as a skill.

## The full install is required

`analyze` and `auto` refuse to run (exit code 3) until every part is installed and its model is on disk: ffmpeg,
demucs + PyTorch, the De-limiter, Apollo and faster-whisper. `bash scripts/setup.sh` installs them all and downloads
the models; `check` must then end with `[ok] every part of the full pipeline is installed and its model is on disk`.

Why it is enforced: an assistant in a sandbox without PyTorch delivered a song with no stem separation and no codec
repair (the vocal exactly as wide as in the original, the codec's 9-13 kHz holes kept) and never said so. The same
song restored with every part sounded clearly better. Assistants that can install packages did so once they were
told it was required.

- **ChatGPT with code execution and similar sandboxes.** Install everything first: run `setup.sh`, or the `pip
  install` lines in it one by one, and get ffmpeg (a static build, or `imageio-ffmpeg`'s binary on the PATH). If a
  step fails, read the error and fix it; a CPU-only PyTorch is enough.
- **If an install truly fails** (no internet at all, a package with no build for this platform): tell the user which
  part failed and what the result will lack (`check` prints both). Only with their agreement run with
  `--degraded "<what failed; the user agreed>"`. `review.md` then lists every missing step, and the delivery repeats
  it. Never use `--no-stems` or `--codec-repair off` to save time.
- **No internet for the studio version** (the lookup failed, but the parts are installed): ask the user for the
  studio version, a copy of the track or any 30 s of it, and build the reference first:
  > `profile --name song --files studio.m4a --song --out song.json`, then
  > `auto my.m4a --work out --content music --music-profile song.json`.

## Long jobs

Codec repair runs at about 6x real time on the CPU: a 5 min song takes about 30 min. It uses an NVIDIA or Apple
silicon GPU automatically when there is one (`check` shows which), and is then much faster.

- Assistants with short tool time-outs: start `analyze` or `auto` in the background
  (`nohup python3 scripts/audiofix.py auto ... > run.out 2>&1 &`) and follow `WORK/log.txt`, which prints every
  10 s chunk with the time left.
- If the command is cut off anyway, run the same command again. Every finished chunk is saved in
  `WORK/apollo_chunks/`, and the run resumes from the last one. `input_repaired.wav` is cached once complete, so a
  rerun does not repeat it. `commands.jsonl` shows a command that was cut off (a start with no end).

## Edits need their reasons

`restore --set key=value` needs `--why "<the measurement, WARN or listener's words>"`, and so does a hand edit of
`plan.json`: every setting that differs from `plan_default.json` must have a reason, or `restore` stops (exit code 2).
The reasons are kept in `plan_used.json` (`edit_reasons`) and listed in `review.md`.

## The run's record

After delivering, `bundle --work out` packs `out/run_record.zip`: every command, the diagnosis, the plans with the
edit reasons, the step ledger, the checks, the review, the skill version and the machine's parts. No audio. When the
user wants a result compared with another assistant's, the two records go into
`compare THEIRS.zip MINE.zip --names X,Y`, which names the first place they part.

## Chat-only assistants (no code execution)

Run the tool yourself and let the assistant interpret:

1. On your computer: `bash scripts/setup.sh`, then
   `python3 scripts/audiofix.py analyze my_file --work out`.
2. Paste `out/diagnosis.md` and `out/plan.json` into the chat with `references/diagnosis.md` and
   `references/tuning.md`, and ask for a review of the plan.
3. Apply the suggested edits, with the assistant's reason:
   `python3 scripts/audiofix.py restore --work out --set ... --why "..."`.
4. Paste `out/verification.md` back. Listen to `out/AB.wav` and describe what you hear; the assistant maps your words
   to knobs with `references/tuning.md`.

## Instructions for a custom GPT, Gem or project

Paste this as the system or project instructions, and attach `SKILL.md` and the `references/` files as knowledge:

```
You restore low-quality live recordings (songs, MC/stage talk, or both) using the "live-audio-restore" skill.
Follow SKILL.md. First install everything (bash scripts/setup.sh) until `audiofix.py check` ends with [ok]; the
skill refuses to run otherwise, and you never work around that without the user's agreement (--degraded). Then:
understand the material, choose a reference profile (for a song, its own studio version: auto finds it when online;
offline, ask the user to upload it), run analyze, read diagnosis.md, review plan.json, run restore, read
verification.md and review.md, deliver restored.wav + AB.wav with a plain-language explanation that answers every
review item, then run `audiofix.py bundle --work out` and keep run_record.zip. Every edit needs its reason:
`restore --set ... --why "..."`. When the listener says something sounds wrong, run `audiofix.py investigate --work
out --complaint "<their words>"` and send its investigate/ABC.wav instead of guessing a knob. You cannot hear audio: every claim must come from
the verification numbers or the user's ears. Never use neural denoisers on live speech, never add fake stereo to
mono, never pick a loudness target the limiter can only reach by squashing. If you cannot run code, ask the user
to run the commands and paste the reports.
```

## Running it with no assistant at all

```bash
bash scripts/setup.sh
python3 scripts/audiofix.py auto song.m4a --work out --content music   # finds the song's studio version online
# read out/diagnosis.md and out/verification.md, listen to out/AB.wav, then tweak:
python3 scripts/audiofix.py restore --work out --set music.match.hi_db=3 --why "too boomy"
```
