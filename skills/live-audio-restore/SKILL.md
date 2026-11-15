---
name: live-audio-restore
description: >-
  Diagnose and restore low-quality live recordings to near-studio quality: concert songs, spoken MC / stage
  talk, or both in one file (audience or phone recordings, bootlegs, TV/radio broadcast rips, Bilibili/YouTube
  rips). It measures what is wrong first (over-limiting, anti-phase layers, muddy or harsh tone, codec band
  limit, hum, clipping, crowd louder than the talk), writes an editable plan that states the evidence for every
  step, renders it (stem separation, neural de-limiting, polarity repair, tonal match to reference recordings,
  stem dynamics, band extension, crowd taming, loudness) and verifies the result with objective checks and a
  loudness-matched A/B file. Use it whenever someone wants to fix, restore, clean up, remaster or improve the
  sound of a live recording or a talk clip, even if they only say it sounds bad, muddy, harsh, squashed, thin,
  phasey, too quiet or drowned by the crowd.
compatibility: >-
  Any agent that can run shell commands with Python 3.9+, numpy and scipy. ffmpeg is recommended; the full
  treatment adds demucs+torch, the De-limiter model and faster-whisper (all CPU-friendly). Missing pieces
  degrade gracefully and are reported.
metadata:
  version: "1.0.0"
---

# Live audio restore

Turns a low-quality live recording into the best version the material allows, in two paths that share one
workflow:

- **Music path** (songs): de-limit, repair anti-phase layers, de-ess, restore drum hits, widen, match each
  instrument and then the whole mix to the tone of reference recordings, control dynamics per instrument, rebuild
  the top octave above a codec cut-off, then master to a loudness the limiter can reach without squashing.
- **Speech path** (MC, stage talk, announcements): keep the voice and push the room underneath, gentle stationary
  denoise, tone toward the long-term speech spectrum, de-ess, compress and level, then hold the crowd down in the
  gaps between sentences without touching the words.
- **Mixed files** (a concert with songs and MCs) are split automatically and each part takes its own path.

Everything is driven by `scripts/audiofix.py`. It always runs in the same order: **analyze → plan → restore →
verify**. The plan is plain JSON, so you or the user can read and change it before anything is rendered.

## Ground rules

1. **You cannot hear the audio.** Every claim about the result must come from `verification.md` numbers or from
   the user's own ears. Never say something "sounds better" on your own authority. Give the before and after
   numbers, and ask the user to listen to `AB.wav`.
2. **Diagnose before fixing.** Each step in the plan carries the measurement that switched it on. If a step looks
   wrong for this material, change the plan and say why. Don't stack extra effects on top to compensate.
3. **Never overwrite the source.** Work in a work folder. Deliver a lossless WAV, an AAC copy and the A/B file.
4. **Some things look helpful and are harmful:**
   - Neural denoisers on live speech: through a PA in a reverberant hall they remove the speech along with the
     noise.
   - Synthetic stereo on a mono source.
   - Matching the tone of a different genre.
   - A loudness target the limiter can only reach by squashing.

   The pipeline avoids all four by default. Keep it that way unless the user asks for one.
5. Explain results in the user's language, in plain terms: what was wrong, what changed, how big the change is,
   and what could not be fixed.

## Quick start

```bash
bash scripts/setup.sh                       # once: installs/clones what is missing, then runs `check`
python3 scripts/audiofix.py auto INPUT --work WORK            # analyze + restore + verify in one go
```

`INPUT` can be any audio or video file ffmpeg reads. `WORK` collects everything:

| File | What it is |
|---|---|
| `diagnosis.md` / `.json` | What is wrong, with numbers |
| `overview.png` | Spectrogram, level and stereo width over time, with the content map |
| `plan.json` | Every decision with its `why` |
| `restored.wav` | 24-bit / 44.1 kHz, the deliverable |
| `restored.m4a` | AAC 256 kb/s copy |
| `verification.md` / `.json` | Checks and before/after numbers |
| `AB.wav` | Original, then restored, for the same passages, loudness-matched per pair |
| `compare.png` | Long-term spectra against the reference, and before/after spectrograms |
| `log.txt` | Everything the run printed |

## Workflow

### 1. Understand the job

Find out, from the request or the file itself, and ask only what you cannot infer:

- **What it is.** Songs, MC, or both. Who is performing.
- **Where it came from.** Broadcast rip, audience phone recording, soundboard.
- **What the user wants.** Natural or strongly remastered; listening, video or archive.

The **reference** decides what "good" means for the tone, so pick it deliberately:

| Situation | Reference |
|---|---|
| The user has studio tracks of this artist | `profile --name X --files a.flac b.mp3 ...`: the best reference |
| Online, artist known | `profile --name X --itunes-artist "Artist" [--country jp]`: official 30 s previews, about 1 min per 3 clips |
| Neither | Bundled `jrock_band` (guitar band, loud modern master). For other genres, build a profile from any representative studio tracks the user can supply |
| Speech | Bundled `speech_ltass` (long-term average speech spectrum), used automatically |

### 2. Check the environment

```bash
python3 scripts/audiofix.py check
```

| Missing | What degrades |
|---|---|
| ffmpeg | Only WAV in; no AAC copy, no stationary denoise, no declip |
| demucs / torch | No stems: polarity repair, tone match and loudness run on the whole mix; no per-instrument steps; speech uses the mix |
| De-limiter | Over-limited music gets stronger drum-transient shaping instead of neural de-limiting |
| faster-whisper | The MC talker is located by voicing instead of word timings (about 80 % accurate against word timings); `verify` loses the intelligibility check |
| numba | Slower, vectorised approximations |

`setup.sh` installs all of them. On a machine without internet, see `references/other-ai.md`.

### 3. Analyze

```bash
python3 scripts/audiofix.py analyze INPUT --work WORK --music-profile PROFILE \
    [--content music|speech] [--language ja] [--asr-model small]
```

Then read `diagnosis.md` and check, in this order:

1. **Content map.** Are the songs and MCs where they should be?
   - A short MC inside a song set is merged into the songs on purpose (minimum 10 s for speech, 20 s for music).
   - If a whole file is misclassified, rerun with `--content`, or edit `segments` in `plan.json`.
2. **Music sections.** Cuts are placed only where the stereo image of the recording chain jumps, for example a
   change of broadcast feed. They are not placed at song changes.
3. **Global findings.**
   - Loudness range and peak-to-loudness: is it over-limited?
   - Bandwidth: is there a codec wall?
   - Mid-band anti-phase.
   - Mono loss, clipping, hum.
4. **Music table.** Which stem is anti-phase, who carries the 3-10 kHz harshness, and the tone deviation per band.
5. **Speech.** How loud the gaps are against the speech, voice-to-background ratio, the talker-detection method
   and language, and whether separation kept the voice (`vocal_share`).

`references/diagnosis.md` explains every number, its threshold, what it sounds like and which step it triggers.
Read it the first time you use the skill, and whenever a verdict surprises you.

### 4. Review and edit the plan

`plan.json` holds one block per step, each with a `why`. Edit it directly, or override at render time:

```bash
python3 scripts/audiofix.py restore --work WORK --set music.delimit.depth=1.5 music.widen.enabled=false
```

Common edits:

| Goal | Edit |
|---|---|
| Natural rather than remastered | `music.match.alpha=0.5`, `music.stem_match.enabled=false`, `music.balance.enabled=false` |
| Less low end | `music.match.hi_db=3` |
| Keep more top end | `music.match.lo_db=-4` |
| Less punch processing | `music.delimit.depth=1.5`, `music.transients.max_db=1.5` |
| No synthetic top | `music.band_ext.enabled=false` |
| More air | `music.band_ext.below_ref_db=-2` (louder than the reference; careful) |
| Spoken parts louder or quieter against the songs | `master.speech_offset_db=2` |
| Keep more audience between sentences | `speech.crowd.cap_db=-2` or `speech.crowd.enabled=false` |
| Louder or more dynamic master | `master.over1_max_pct` and `master.over2_max_pct` (8 and 1.5 by default) |

### 5. Restore and verify

```bash
python3 scripts/audiofix.py restore --work WORK      # renders, then runs verify
python3 scripts/audiofix.py verify --work WORK       # re-check an existing render
```

### 6. Read the verification and act on warnings

Every check is PASS or WARN, with before and after numbers. Investigate each WARN before delivering:

| WARN | Usual cause | Action |
|---|---|---|
| Music not closer to the reference tone | Wrong genre profile, or a very short music part | Build an artist profile; or lower `music.match.alpha` |
| Dynamics squashed further | Loudness target too high for this material | Lower `master.over1_max_pct` / `over2_max_pct` |
| Anti-phase left in the mids | An anti-phase layer in a stem that was not flagged | Add the stem to `music.antiphase.stems` |
| Mono loss | Same, or too much width | Lower `music.widen.side_db` |
| Crowd still competes with the talk | Talker found by voicing, not words | Install faster-whisper; or `speech.crowd.cap_db=4`, `floor_db=24` |
| Talker's words recognised worse | Something removed speech (denoise, EQ) | Lower `speech.denoise.nr_db`; check `speech.source` |
| True peak above the ceiling | Should not happen | Report it; rerun `restore` |

`compare.png` shows the long-term spectrum of input, output and reference. The output should hug the reference
curve below the input's cut-off. Above the cut-off, the rebuilt top should continue the reference slope.

### 7. Deliver

Send `restored.wav` (lossless), `restored.m4a` (for phones) and `AB.wav`, and ask the user to listen to `AB.wav`
first. In the message:

- What was wrong, in plain words, with the key numbers.
- What each step did, with before and after numbers from `verification.md`.
- What was not fixed or cannot be fixed: invented content, crowd under the words, codec holes.
- Which knobs exist if they want it different.

Do not describe a change you did not measure.

### 8. Iterate with the listener

Map what the user says to a knob, change one thing at a time, re-render and send a new `AB.wav`. For example:

- "Muddy" → less 200-400 Hz.
- "Harsh" → stronger de-ess, or less 3-8 kHz.
- "Thin" → more 60-120 Hz.
- "Squashed" → a lower loudness target.
- "Phasey" → check the anti-phase stems.
- "Too much effect" → toward natural.

`references/tuning.md` has the full table for both paths.

## The two paths in brief

**Music** (`references/music-path.md`):

1. Repairs on the raw mix: 20 Hz high-pass, dehum, declip.
2. De-limit: the network's gain curve at 1.5-2x depth, on every stem.
3. Polarity repair of anti-phase stems.
4. De-ess the vocal; lift drum transients; widen the guitar/keys layer.
5. Per-stem tone match, 60 % toward the reference stems, per section.
6. Stem balance, halfway to the reference ratios.
7. Shave drum peaks on the drum stem; compress bass, vocal and guitars separately; light bus glue.
8. Overall tone match, 85 %, per section.
9. Band extension above the codec cut-off.
10. Bass below 100 Hz to mono.
11. Perceptual loudness and a true-peak limiter at -1 dBTP.

**Speech** (`references/speech-path.md`):

1. Locate the talker: Whisper word timings, or voicing as a fallback.
2. Voice from the vocal stem, the rest underneath at -6 to -12 dB. If separation lost the voice, work on the mix.
3. Stationary denoise of 6-10 dB (ffmpeg afftdn; never a neural denoiser).
4. High-pass at 75 or 100 Hz.
5. Half-way tone match to the speech spectrum, boosts at most +3 dB.
6. De-ess, 2.5:1 compression, slow leveler.
7. Crowd taming in the gaps; speech frames move at most 1.5 dB.
8. Loudness: -16 LUFS for speech only, 4 dB under the music in a concert.

**Mixed files.** Content classes are joined with 1 s crossfades. Gaps take the class of their neighbours.

## Using this skill outside Claude

The skill is plain Markdown plus a Python command-line tool, so any assistant or person that can run commands
can follow it: Codex, ChatGPT with code execution, Gemini, Cursor, Copilot, or a human. `AGENTS.md` at the
repository root points agents here. `references/other-ai.md` covers chat-only assistants, sandboxes without
internet, and a prompt to paste into a custom GPT or project.

## Files

```
SKILL.md                     this workflow
scripts/audiofix.py          CLI: check | profile | analyze | restore | verify | auto
scripts/audiofix/            the library (analysis, plan rules, music, speech, master, verify, DSP, loudness)
scripts/setup.sh             installs dependencies and the De-limiter model
scripts/make_test_clip.py    synthetic degraded song / MC clips with known problems (tests, demos)
assets/profiles/             bundled references: jrock_band.json, speech_ltass.json
references/diagnosis.md      how to judge: every measurement, threshold, symptom and decision, both paths
references/music-path.md     each music step: why, how, parameters, failure modes
references/speech-path.md    each speech step: why, how, parameters, failure modes
references/verification.md   checks, A/B listening protocol, what the plots should look like
references/tuning.md         listener feedback -> plan edits, both paths
references/pitfalls.md       mistakes already made once, and how the pipeline avoids them
references/other-ai.md       ChatGPT / Codex / Gemini / offline use
evals/                       test cases (prompts, expected outcomes, synthetic inputs)
```
