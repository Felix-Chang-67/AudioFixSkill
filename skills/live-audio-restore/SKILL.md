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
license: MIT (see LICENSE.txt)
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
- **Speech path** (MC, stage talk, announcements): the voice gets the music path's vocal treatment. It is EQ'd toward
  the singer's restored vocal sound, compressed like a vocal, and de-limited and band-extended when measured. The room
  is kept underneath, with gentle stationary denoise and a leveler, and the crowd is held down in the gaps between
  sentences without touching the words.
- **Mixed files** (a concert with songs and MCs) are split automatically and each part takes its own path.

**The user's word beats detection.** When the user says what a file is, pass `--content music` for a song or
`--content speech` for an MC or talk. Leave the default `auto` only for a file that holds both, or one nobody
described. Detection can misfire: a quiet sung passage without drums looks like talk, because singing pauses
between phrases and Whisper writes the lyrics down as words.

Everything is driven by `scripts/audiofix.py`. It always runs in the same order: **analyze → plan → restore →
verify**. The plan is plain JSON, so you or the user can read and change it before anything is rendered.

## Ground rules

1. **You cannot hear the audio.** Every claim about the result must come from `verification.md` numbers or from
   the user's own ears. Never say something "sounds better" on your own authority. Give the before and after
   numbers, and ask the user to listen to `AB.wav`.
2. **Diagnose before fixing.** Each step in the plan carries the measurement that switched it on. If a step looks
   wrong for this material, change the plan and say why. Don't stack extra effects on top to compensate.
3. **Trust the defaults; edit only for a reason.** A reason is a WARN, a measurement the rules missed, or the
   listener's words.
   - The rules already encode what was learned on real recordings: limited or open dynamics, a whole reversed layer
     or anti-phase in a few bands, a mixed or a room-miked stereo image.
   - Two assistants hand-tuning the same song with this skill delivered results 3 dB apart in loudness and 3 LU apart
     in dynamics. Write every edit and its reason into the delivery message.
4. **Never overwrite the source.** Work in a work folder. Deliver a lossless WAV, an AAC copy and the A/B file.
5. **Some things look helpful and are harmful:**
   - Neural denoisers on live speech: through a PA in a reverberant hall they remove the speech along with the
     noise.
   - Synthetic stereo on a mono source, or extra width on a room-miked recording that is already wide.
   - Matching the tone of a different genre.
   - A loudness target the limiter can only reach by squashing.
   - Flattening an open recording: its quiet intros and breaks are part of the performance.
   - A "safe" result quieter than the original: it reads as "nothing changed".

   The pipeline avoids all of these by default. Keep it that way unless the user asks for one.
6. Explain results in the user's language, in plain terms: what was wrong, what changed, how big the change is,
   and what could not be fixed.
7. **Work carefully: all checks PASS is not the end.** The tool prints `[review]` lines after `analyze` and after
   `restore`, and writes `review.md`. They name what a careful engineer would check, from the measurements:
   - applause or clapping in the file;
   - a missing rhythm section or bass, which means a different arrangement from the studio version;
   - extreme peaks;
   - a weak or generic tone reference;
   - rebuilt highs;
   - an EQ or balance step that ran into its limit;
   - the edits you made to the default plan.

   Answer every one in the delivery. Assistants that stopped at PASS shipped applause raised by 6 dB and a vocal
   lowered by 4.4 dB.

## Quick start

```bash
bash scripts/setup.sh                       # once: installs/clones what is missing, then runs `check`
python3 scripts/audiofix.py auto INPUT --work WORK --content music    # a song: analyze + restore + verify
python3 scripts/audiofix.py auto INPUT --work WORK --content speech   # an MC / talk
python3 scripts/audiofix.py auto INPUT --work WORK                    # songs and MCs in one file: split automatically
```

For a song, `auto` also looks up the studio version of that very song online, the best tone target (step 1). It
takes the artist and title from the file's tags, or from `--artist` / `--title`.

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
| `vocal_eq.json` | Songs only: the EQ given to the vocal stem. MCs from the same show can reuse it |
| `review.md` | What to check and answer before delivering, with the cause and knob of every WARN |
| `steps.json` | The step ledger: loudness, 3 s crest, loudness range, mono loss and tone after every music step |
| `previous/` | The render before the last one, with its plan and checks, for comparison |
| `log.txt` | Everything the run printed |

## Workflow

### 1. Understand the job

Find out, from the request or the file itself, and ask only what you cannot infer:

- **What it is.** Songs, MC, or both. Who is performing. This sets `--content`: `music` for a song, `speech` for
  an MC or talk, the default `auto` only for a file with both. A file name or a request like "fix this song" is
  enough to go on.
- **Where it came from.** Broadcast rip, audience phone recording, soundboard.
- **What the user wants.** Natural or strongly remastered; listening, video or archive.

The **reference** decides what "good" means for the tone, so pick it deliberately:

| Situation | Reference |
|---|---|
| A single song, and its studio version exists | **Automatic** when the tags or `--artist` / `--title` name the song and the network allows: `analyze` fetches the official iTunes preview, checks that its passage is in the recording and measures tone there, like with like. Without network, ask the user for the studio track: `profile --name X --files track --song --out X.json`, then `--music-profile X.json`. **The best reference**, it is what "like the album" means: on an audience recording it brought the mix within 1.7 dB RMS of the studio master in that passage. Live, remix and similar versions are skipped |
| The user has studio tracks of this artist | `profile --name X --files a.flac b.mp3 ...` |
| Online, artist known | `profile --name X --itunes-artist "Artist" [--country jp]`: official 30 s previews, about 1 min per 3 clips |
| Neither | Bundled `jrock_band` (guitar band, loud modern master). For other genres, build a profile from any representative studio tracks the user can supply |
| Speech | Bundled `speech_ltass` (long-term average speech spectrum), used automatically |

`analyze` logs which reference it used and why (`[reference]` lines and `diagnosis.md`). If a song ended up on the
generic profile, a `[next]` line says so. Ask the user for the studio version rather than delivering silently.
`--music-profile` chooses the reference yourself; `--no-lookup` keeps everything offline.

### 2. Check the environment

```bash
python3 scripts/audiofix.py check
```

| Missing | What degrades |
|---|---|
| ffmpeg | Only WAV in; no AAC copy, no stationary denoise, no declip |
| demucs / torch | No stems: polarity repair, tone match and loudness run on the whole mix; no per-instrument steps; speech uses the mix. **Songs and talk can no longer be told apart**: the file is treated as music unless you pass `--content speech` (an MC) |
| De-limiter | Over-limited music gets stronger drum-transient shaping instead of neural de-limiting |
| faster-whisper | The MC talker is located by voicing instead of word timings (about 80 % accurate against word timings); `verify` loses the intelligibility check |
| numba | Slower, vectorised approximations |
| Apollo (codec repair) | On low-bitrate rips (cut-off below 16 kHz and a sparse band under it), the highs keep the codec's sparse, metallic texture: "fake", harsh air. It runs on an NVIDIA (CUDA) or Apple silicon (MPS) GPU when one is found, and falls back to the CPU otherwise or if the GPU run fails. On the CPU it is slow, about 6x real time, so for long concerts you may choose `--codec-repair off` on purpose. `check` shows the device; `AUDIOFIX_DEVICE=cpu|cuda|mps` forces one |

`setup.sh` installs all of them. On a machine without internet, see `references/other-ai.md`.

### 3. Analyze

```bash
python3 scripts/audiofix.py analyze INPUT --work WORK [--content music|speech] \
    [--music-profile PROFILE | --artist A --title T | --no-lookup] [--language ja] [--asr-model small]
```

Then read the `[review]` lines and `diagnosis.md`, and check, in this order:

0. **Timeline.** What is in the file, per 15 s: level, peaks, and each stem's share. Look for the intro, quiet
   passages, applause, talk, and instruments that are missing.
1. **Content map** (with `auto` only). Are the songs and MCs where they should be?
   - A short MC inside a song set is merged into the songs on purpose (minimum 10 s for speech, 20 s for music).
   - A song with "speech" in it, or an MC with "music", is a misclassification: rerun with `--content`, or edit
     `segments` in `plan.json`. Quiet sung passages without drums are the usual victims.
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

`plan.json` holds one block per step, each with a `why`. Read the `why` of these five decisions before rendering;
they matter most:

1. **Limited or open** (`music.delimit`, `music.dynamics`, `music.macro`, `master`):
   - A limited source (LRA < 4 LU, or PLR < 11 dB with a 3 s crest under 12.5 dB) is de-limited, gets per-stem
     peak control and a loud master.
   - An open one keeps its dynamics: light compression, the original's loudness contour put back, and a limiter
     that stays almost idle.
2. **Anti-phase** (`music.antiphase.stems`): a whole reversed layer, or anti-phase confined to some bands. Only
   the anti-phase bins are rotated, so a false positive costs little and a miss costs mono compatibility.
3. **Width** (`music.widen`): only on a mixed (board or broadcast) image, never on a room-miked one.
4. **Codec wall** (`music.band_ext`): the top above the cut-off is rebuilt.
5. **Reference** (`profiles.music`): the target for every tone decision.

Edit the plan directly, or override at render time:

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
| Lighter, more natural voice treatment on MCs | `speech.mode=gentle` |
| MC from a show whose songs were restored separately | `speech.match.target=song_vocal speech.match.chain_file=SONGWORK/vocal_eq.json` |
| Louder or more dynamic master | `master.over1_max_pct` and `master.over2_max_pct`: 8 and 1.5 for limited sources, 2.5 and 0.5 for open ones |
| Denser, more "produced" sound on an open recording | `music.macro.enabled=false`; stronger `music.dynamics` ratios |

### 5. Restore and verify

```bash
python3 scripts/audiofix.py restore --work WORK      # renders, then runs verify
python3 scripts/audiofix.py verify --work WORK       # re-check an existing render
```

### 6. Read the verification and the review, and act on warnings

Every check is PASS or WARN, with before and after numbers. `review.md` traces each WARN to the step that moved its
measure (the ledger in `steps.json`) and names that step's knob.

- Change that one knob, re-render, and compare. The previous render stays in `previous/`, and `review.md` lists what
  changed between the two.
- Do not change several knobs at once. Do not run control renders by hand: the ledger already shows which step did
  what.

The usual causes:

| WARN | Usual cause | Action |
|---|---|---|
| Music not closer to the reference tone | Wrong genre profile, or a very short music part | Build an artist profile; or lower `music.match.alpha` |
| Dynamics squashed further | Loudness target too high for this material, or the music reached the master already compressed (the `[master]` log says so) | Lower `master.over1_max_pct` / `over2_max_pct`; in the second case lighten `music.dynamics` |
| Quiet and loud parts not kept apart (open recording) | Compression or a hand edit flattened the song | Check that `music.macro` is on and `music.dynamics` is light; lower the master criteria |
| Quieter than the original (open recording) | Master criteria set too strict by hand | Raise `master.over1_max_pct` / `over2_max_pct` toward 2.5 / 0.5 |
| Anti-phase left in the mids | An anti-phase layer in a stem that was not flagged | Add the stem to `music.antiphase.stems` |
| Mono loss | Same, or too much width | Lower `music.widen.side_db` |
| Crowd still competes with the talk | Talker found by voicing, not words | Install faster-whisper; or `speech.crowd.cap_db=4`, `floor_db=24` |
| Talker's words recognised worse | Something removed speech (denoise, EQ) | Lower `speech.denoise.nr_db`; check `speech.source` |
| True peak above the ceiling | Should not happen | Report it; rerun `restore` |

`compare.png` shows the long-term spectrum of input, output and reference. The output should hug the reference
curve below the input's cut-off. Above the cut-off, the rebuilt top should continue the reference slope.

### 7. Deliver

Send `restored.wav` (lossless), `restored.m4a` (for phones) and `AB.wav`, and ask the user to listen to `AB.wav`
first. The last `[next]` lines of `restore` repeat this. In the message:

- What was wrong, in plain words, with the key numbers.
- An answer to every item of `review.md`: what it is, what was done, and what the listener may want.
- Which reference the tone was matched to: the studio version of this song, or the generic profile and why.
- What each step did, with before and after numbers from `verification.md`.
- What was not fixed or cannot be fixed: invented content, crowd under the words, codec holes.
- Every edit you made to the plan, with its reason.
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

1. Repairs on the raw mix: 20 Hz high-pass, dehum, declip. On low-bitrate rips, codec repair (Apollo) first
   refills the sparse band under the cut-off.
2. De-limit (limited sources): the network's gain curve at 1.5-2x depth, on every stem.
3. Polarity repair of anti-phase stems: whole reversed layers, or anti-phase in some bands only.
   On room-miked images, vocals, drums and bass are narrowed toward the studio stems' width.
4. De-ess the vocal; lift drum transients; widen the guitar/keys layer (mixed images only).
5. Per-stem tone match, 60 % toward the reference stems, per section.
6. Stem balance, halfway to the reference ratios.
7. Shave drum peaks on the drum stem; compress bass, vocal and guitars separately; bus glue. Firm on limited
   sources, light on open ones.
8. Overall tone match, 85 %, per section.
9. Band extension above the codec cut-off, half of it noise-like so the rebuilt air is dense, not metallic. It adds
   only what is missing, per 20 s and 250 Hz, up to the reference's own level, so the top follows the studio roll-off.
10. Bass below 100 Hz to mono.
11. Open recordings: the original's loudness contour put back, so quiet parts stay quiet.
12. Perceptual loudness and a true-peak limiter at -1 dBTP: as loud as a modern master for limited sources; for
    open ones the limiter stays almost idle.

**Speech** (`references/speech-path.md`). The voice gets the music path's vocal treatment (`speech.mode` v2; `gentle`
is the earlier, lighter path):

1. Locate the talker: Whisper word timings, or voicing as a fallback.
2. Voice from the vocal stem, the rest underneath at -6 to -12 dB. If separation lost the voice, work on the mix.
3. De-limit when the songs of the same file show the broadcast limiter and the talk's PLR is low.
4. Stationary denoise of 6-10 dB (ffmpeg afftdn; never a neural denoiser).
5. High-pass at 75 or 100 Hz.
6. Voice EQ, 85 % toward the target, with boosts only where the voice dominates (+3 / -8 dB). The target is the first
   that applies:
   - the songs' vocal-stem EQ when the file has songs;
   - otherwise the profile's studio vocal above 400 Hz with the speech spectrum below;
   - otherwise the speech spectrum.
7. De-ess, the music path's vocal compressor (3:1), slow leveler.
8. Crowd taming in the gaps; speech frames move at most 1.5 dB.
9. Band extension above a codec wall, 3 dB gentler than for music.
10. Loudness: -16 LUFS for speech only, 4 dB under the music in a concert.

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
