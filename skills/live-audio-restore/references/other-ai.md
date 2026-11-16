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

## ChatGPT with code execution (Advanced Data Analysis / Code Interpreter) and similar sandboxes

These sandboxes usually have numpy, scipy and matplotlib, but **no internet**, often no ffmpeg, and no
torch/demucs/numba.

1. Upload the repository as a zip, and the audio as **WAV** (no ffmpeg means no m4a/mp3 decoding).
2. Tell the assistant:
   > Unzip the repo, read `skills/live-audio-restore/SKILL.md` and follow it. Run
   > `python3 skills/live-audio-restore/scripts/audiofix.py check`, then
   > `auto my.wav --work out --no-stems` (add `--content speech` if it is an MC / talk). Explain
   > `out/diagnosis.md` and `out/verification.md` to me and give me `out/restored.wav` and `out/AB.wav`.

   Without stems, songs and talk cannot be told apart automatically: the drum and bass stems are what separates
   them, and Whisper alone cannot, because lyrics are words too. Tell the assistant which it is.
3. What still works without stems, models and numba:
   - all measurements and the plan;
   - polarity repair and tone match on the whole mix;
   - band extension;
   - the perceptual loudness and limiter (vectorised, slower);
   - verification and A/B.

   Without stems there is no per-instrument work. Speech uses the voicing detector and the whole mix. Expect a
   clearly smaller improvement than the full pipeline: run the full version locally when you can.

   Codec repair (Apollo) is missing there too. On low-bitrate rips, the rebuilt highs then keep the codec's sparse,
   metallic texture. The noise-like half of the band extension softens this, but cannot replace the repair.

## Long jobs

Codec repair runs at about 6x real time on the CPU: a 5 min song takes about 30 min. It uses an NVIDIA or Apple
silicon GPU automatically when there is one (`check` shows which), and is then much faster. Assistants with short tool
time-outs should start `analyze` in the background, or ask the user to run it, and continue from the work folder.
`input_repaired.wav` is cached, so a rerun does not repeat it.

## Chat-only assistants (no code execution)

Run the tool yourself and let the assistant interpret:

1. On your computer: `bash scripts/setup.sh`, then
   `python3 scripts/audiofix.py analyze my_file --work out`.
2. Paste `out/diagnosis.md` and `out/plan.json` into the chat with `references/diagnosis.md` and
   `references/tuning.md`, and ask for a review of the plan.
3. Apply the suggested edits:
   `python3 scripts/audiofix.py restore --work out --set ...`.
4. Paste `out/verification.md` back. Listen to `out/AB.wav` and describe what you hear; the assistant maps your words
   to knobs with `references/tuning.md`.

## Instructions for a custom GPT, Gem or project

Paste this as the system or project instructions, and attach `SKILL.md` and the `references/` files as knowledge:

```
You restore low-quality live recordings (songs, MC/stage talk, or both) using the "live-audio-restore" skill.
Follow SKILL.md: understand the material, choose a reference profile, run analyze, read diagnosis.md, review
plan.json, run restore, read verification.md, deliver restored.wav + AB.wav with a plain-language explanation,
then iterate on listener feedback with references/tuning.md. You cannot hear audio: every claim must come from
the verification numbers or the user's ears. Never use neural denoisers on live speech, never add fake stereo to
mono, never pick a loudness target the limiter can only reach by squashing. If you cannot run code, ask the user
to run the commands and paste the reports.
```

## Running it with no assistant at all

```bash
bash scripts/setup.sh
python3 scripts/audiofix.py auto concert.m4a --work out --music-profile jrock_band
# read out/diagnosis.md and out/verification.md, listen to out/AB.wav, then tweak:
python3 scripts/audiofix.py restore --work out --set music.match.hi_db=3
```
