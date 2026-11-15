# How to judge what to fix

`analyze` runs every measurement below and writes the verdicts into `diagnosis.md`. `plan.json` follows these rules.
This document explains each number, so you can tell the user what it means and override a rule when the material
calls for it.

## Contents

1. [From what you hear to what to measure](#1-from-what-you-hear-to-what-to-measure)
2. [Global measurements (both paths)](#2-global-measurements-both-paths)
3. [Content map: songs, MC, gaps, sections](#3-content-map-songs-mc-gaps-sections)
4. [Music path: decisions](#4-music-path-decisions)
5. [Speech path: decisions](#5-speech-path-decisions)
6. [Worked examples](#6-worked-examples)
7. [When the numbers and the ears disagree](#7-when-the-numbers-and-the-ears-disagree)

---

## 1. From what you hear to what to measure

People describe problems in words. These are the measurements that usually sit behind each word.

| What people say | Look at | Typical cause in live material | Step that fixes it |
|---|---|---|---|
| Flat, squashed, no punch, tiring | LRA < 4 LU, PLR < 11 dB, 3 s crest < 12 dB | Broadcast or streaming limiter | De-limit; drum transients |
| Phasey, hollow, "outside my head", vanishes on a phone speaker | Negative mid-band L-R correlation, mono loss > 1.5 dB, a stem with anti-phase energy | One side of a stereo pair wired in reverse; a widener; a matrix (Lt/Rt) downmix | Polarity repair (per stem) |
| Muddy, boxy | 200-500 Hz above the reference | Room, PA, close mics | Tone match (per stem, then overall) |
| Harsh, shrill, fizzy | 3-10 kHz above the reference; vocal carries most of that energy | Broadcast presence boost, cymbal bleed, crowd | De-ess; tone match |
| Thin, no low end | 60-120 Hz below the reference; bass stem far below its usual share | Small PA, audience position, bass buried | Stem balance; tone match (bass and kick are never thinned) |
| Dull, closed, "like a phone" | Brick-wall cut-off below 18.5 kHz | Lossy codec (AAC/MP3 at 96-192 kb/s) | Band extension (music) |
| Humming, buzzing | Narrow peaks at 50/60 Hz and harmonics | Ground loop | Notch filters |
| Crackling, distorted on loud hits | Flat-topped sample runs | Recorder overload | Declip |
| Can't follow the talk, the crowd is louder than the MC | Gap level relative to speech > -8 dB | Audience mics, cheering between sentences | Crowd taming |
| Talk is boomy or muffled | Rumble below 80 Hz; voice deviates from the speech spectrum | Room, PA | High-pass; speech tone match |
| Noisy hiss under the talk | Voice-to-background < 15 dB | Distant recorder | Stationary denoise (gentle) |
| Too quiet | Integrated loudness far under the target | Recorded low | Loudness stage |

## 2. Global measurements (both paths)

All levels are taken on the decoded 44.1 kHz stereo signal.

### Loudness, loudness range, true peak, PLR, crest

| Measure | What it is | How to read it |
|---|---|---|
| Integrated loudness (LUFS) | ITU-R BS.1770, gated | Masters: about -7 to -10. Broadcast music: about -9 to -12. Raw audience recordings: often -25 to -45. Podcast/speech target: -16 |
| Loudness range (LRA, LU) | EBU Tech 3342: spread of the 3 s loudness | **< 4 LU = limited, < 2 LU = heavily limited.** Natural live music: 4-10 LU. A broadcast rip measured 0.65 LU. MC files run 15-30 LU (the crowd jumps in level) |
| True peak (dBTP) | 4x oversampled peak | Above -1 dBTP risks clipping after lossy encoding |
| PLR (dB) | True peak minus integrated loudness | **< 11 dB = limited, < 9.5 dB = heavily limited.** Loud modern masters: about 8-9. Open live mixes: 12-16 |
| 3 s crest (dB) | Median over 3 s windows of peak / RMS | How much hits stand out locally. Below about 11 dB, punch is gone |

**Rule (music):** limited if LRA < 4 or PLR < 11. If the De-limiter is installed, de-limit at depth 2.0 when LRA < 2 or
PLR < 9.5, otherwise at depth 1.5. Without the De-limiter, drum transients get stronger shaping instead.

### Clipping

The measure counts runs of two or more identical samples within 0.3 dB of the file's peak, per minute.

- **20 or more per minute = digital clipping.** If ffmpeg is available, the mix goes through ffmpeg `adeclip` before
  anything else.
- Lossy codecs smear flat tops, so a broadcast rip of a clipped source can read low. If the user hears crackle on
  loud hits and the count is low, force it with `--set repair.declip.enabled=true`.
- A broadcast limiter piles samples up near its ceiling without making flat runs. That is limiting, not clipping.

### Mains hum

The measure takes the peak at 50 or 60 Hz and its first six harmonics, minus the median of the surrounding spectrum.

- **Hum is present if the fundamental stands 10 dB or more above its surroundings, or two harmonics stand 8 dB or
  more above theirs.**
- The fix is notch filters (Q 35) on every harmonic at 6 dB or more.
- Small excesses (2-6 dB) are normal. Bass notes land on those frequencies too.

### Bandwidth and codec notches

The measure finds the highest frequency within 45 dB of the 1-4 kHz level, and the drop across that edge.

- **A brick wall is an edge drop above 25 dB at a cut-off below 20 kHz.** That is a lossy codec's low-pass:
  AAC/MP3 at 128-192 kb/s cut at about 15-17 kHz, 96 kb/s at about 11-14 kHz.
- For music, a wall below 18.5 kHz switches on band extension. A natural roll-off (small edge drop) is left alone.
- Codec notches are bands of more than 150 Hz that sit at least 4 dB below the smoothed spectrum, for example
  6.4-7.8 kHz at 7.9 dB. They are reported but not filled: a lossy codec has thrown that information away.
  The overall tone match smooths them in the long-term average.

### Stereo

| Measure | Meaning |
|---|---|
| Side minus mid (dB) | Below -45: mono; below -25: near-mono; else stereo. Typical stereo mixes: -6 to -12. Around 0 or above means something is anti-phase |
| L-R correlation per band | +1 identical, 0 unrelated, -1 opposite. Healthy mixes are positive in the bass and mids. 8-15 kHz near 0 is normal for lossy codecs (noise-filled highs) |
| Mid-band (300 Hz-3 kHz) correlation over 4 s windows: worst value, mean, share negative | **Negative in 10 % or more of windows = anti-phase problem** |
| Mono loss (dB) | Loudness of (L+R)/2 minus stereo loudness. **Below -1.5 dB = poor mono compatibility.** Phone speakers and Bluetooth mono drop the cancelled parts |

**Why per stem.** An anti-phase layer is usually one instrument: a stereo-miked guitar amp with one mic wired in
reverse, or a widener effect. The vocal must not be "fixed" along with it. So the decision is taken per separated
stem (next section). The mix-level numbers only confirm that the problem exists.

### Noise floor

The 5th and 50th percentiles of 100 ms RMS. A small spread means the material never goes quiet: crowd, broadcast
compression. This is informational; speech denoise is driven by the voice-to-background ratio.

## 3. Content map: songs, MC, gaps, sections

With stems available, each second gets a label from 3 s windows, smoothed over 9 s.

| Label | Rule |
|---|---|
| music | Drums+bass share of the energy at 0.08 or more, or nothing points to speech |
| speech | Rhythm share below 0.08 and either (a) Whisper finds talk-like word density, at least 0.6 words/s over 9 s, or (b) the vocal stem is at least 15 % of the energy and **pauses like talking**: at least 25 % of its 10 ms frames sit 20 dB under the window's peak |
| gap | Near silence (35 dB below the loud parts), or crowd noise with neither band nor voice |

**Why pauses.** Singing is continuous: the median pause share on live songs is about 0.05. Talk pauses between
phrases: about 0.45. Rhythm share alone fails on breakdowns and ballads. Syllable-rate modulation did not separate the
two on live material.

**Why Whisper word density.** Separation can lose a voice entirely: robotic, heavily reverberant or crowd-masked
voices end up in "other". Words still show up in the mix. Crowd shouts are sparse and talk is dense (2-4 words/s).

**Merging.**

- Minimum lengths are 20 s for music, 10 s for speech and 2 s for gaps. Shorter runs join their longer neighbour,
  so a bass player noodling during an MC does not become a "song", and a sung breakdown does not become an "MC".
- For processing, a gap takes its neighbours' class. It is split at its middle when the neighbours differ.
- A file with neither music nor speech is treated as speech if it has no rhythm section, otherwise as music.

**Sections inside music.**

- A cut is placed where the recording chain changes, seen as a jump of the stereo image over 20 s on either side:
  |change in side-minus-mid| / 4 dB + |change in mid-band correlation| / 0.5 must reach 1.2, at least 30 s from
  other cuts.
- Tonal shape is deliberately not used. It changes with the arrangement, and equalising each part of a song toward
  the same curve would flatten the music.
- Each section gets its own tone match and stem balance, crossfaded over 1.5 s.
- Example: a broadcast whose second half came from another feed was cut at 211 s, where side-minus-mid went from
  -9.7 to -2.4 dB and correlation from 0.81 to -0.10. Song changes produced no cuts.

**When to override.**

- A file that is all MC or all music but was split anyway: use `--content speech` or `--content music`.
- A known feed change the detector missed: add it to `sections` in `plan.json`.

## 4. Music path: decisions

`diagnosis.md` has a stem table (level against the mix, anti-phase energy, worst and median correlation, anti-phase
phase by band), the 3-10 kHz share by stem, and the tone deviation against the music profile by band.

| Step | Switched on when | Strength / parameters |
|---|---|---|
| De-limit | LRA < 4 or PLR < 11, and the De-limiter is installed | Depth 2.0 if LRA < 2 or PLR < 9.5, else 1.5 |
| Drum transients | Stems available | 0.35 / max 2.5 dB with de-limiting; 0.55 / 4.5 dB if limited but no de-limiter; 0.25 / 2 dB otherwise |
| Polarity repair | A stem has 10 % or more of its energy in 4 s windows with L-R correlation below -0.3 | Only that stem. Without stems, the mix when 10 % or more of mid-band windows are negative |
| De-ess (vocal) | Vocal stem present | Max cut 9 dB if the vocal carries 50 % or more of the 3-10 kHz energy, or 5-8 kHz sits more than 4 dB above the profile; else 6 dB. Acts only on the brightest 20 % of moments |
| Widen guitars/keys | Stereo source and stems | Side +3 dB above 150 Hz. Never on mono or near-mono |
| Per-stem tone match | Stems and a profile with stem statistics | 60 % toward the reference stem. Boosts at most +3 dB (+4 dB below 150 Hz), cuts at most -6 dB. Boost only where the stem carries at least 5-30 % of the band (no lifting of bleed). Bass and kick never cut below 120 Hz |
| Stem balance | Same | Each stem halfway to its reference share of the mix, at most ±3 dB, per section |
| Stem dynamics | Stems | Drum peaks held at drum loudness +11 dB (1 ms lookahead); bass 3:1, vocal 3:1, guitars 2.5:1; bus 1.8:1 |
| Overall tone match | Always (with a profile) | 85 % toward the profile, -8 to +6 dB, no boosts above the cut-off, per section |
| Band extension | Brick wall below 18.5 kHz | SBR-style shift of the band below the cut-off, level set so the band above the cut-off matches the profile's top octave |
| Bass to mono | Stereo | Below 100 Hz |
| Loudness | Always | Loudest target where the limiter exceeds 1 dB at most 8 % of the time and 2 dB at most 1.5 % |

**Reading the anti-phase picture.** In the "anti-phase by band" column, cosines near -1 in 150-2500 Hz with +1 above
6 kHz mean a coherent layer reversed only in its body. A plain channel swap would be reversed everywhere. The repair
rotates only coherent anti-phase bins, so this pattern is handled correctly.

In one broadcast, the vocal stem showed 3.4 % anti-phase energy. That was bleed of the guitar layer, below the
10 % threshold, and it was rightly left alone.

**Reading the tone table.** "+" means too much against the reference. A typical broadcast or audience pattern:

| Band | Deviation | Meaning |
|---|---|---|
| 80 Hz | -6 to -9 dB | No weight |
| 315 Hz | +4 to +6 dB | Mud |
| 5-8 kHz | +5 to +10 dB | Harshness |

Check the 3-10 kHz share before blaming cymbals. On the broadcast example, the vocal carried 61-68 % of it.

**Choosing the profile matters more than any knob.** Genre sets the target. A jazz club recording matched to a loud
rock profile will be pushed toward rock. Build a profile from the artist or the genre whenever possible.

## 5. Speech path: decisions

`diagnosis.md` reports, over the spoken parts:

| Measure | Meaning |
|---|---|
| speech_pct | Share of time the talker is speaking |
| gap_minus_speech_db | Median level in the gaps minus median speech level (400 ms windows). **Above -8 dB, the crowd competes with the talk.** Live MCs measured +5 to +12 dB: cheering louder than the talk |
| gap_p90_minus_speech_db | The loudest 10 % of the gaps. Values of +20 dB mean bursts that make listeners reach for the volume knob |
| speech_to_background_db | Vocal stem against everything else during speech. Unknown when separation lost the voice |
| rumble_db | Energy below 80 Hz against 200-800 Hz in the voice. **Above -12 dB = rumble** |
| voice_bandwidth | Where the voice's spectrum ends. Tone matching never boosts above it |
| vocal_share | Share of the spoken parts that separation put into the vocal stem. **Below 0.3, the voice was lost:** the talker is located on the mix and the speech path works on the mix |
| vad | `asr` (Whisper word timings, chosen language with scores) or `harmonic` |

**Locating the talker is the key decision.** Crowd taming lowers everything outside the talker's speech. A detector
that mistakes cheering for speech leaves the crowd untouched; one that misses speech turns words down.

| Method | Accuracy | Notes |
|---|---|---|
| Whisper word timings (`asr`) | Words only, so crowd shouting without words is excluded | Language picked by decoding a sample in the 3 likeliest languages and keeping the most probable. Language ID alone took a synthetic English voice for Welsh with 77 % confidence |
| Voicing (`harmonic`) | About 80 % F1 against word timings; misses part of the crowd | Fallback without faster-whisper |

| Step | Switched on when | Strength / parameters |
|---|---|---|
| Source | Stems available and vocal_share ≥ 0.3 | Voice from the vocal stem; the rest high-passed at 90 Hz underneath at -12 dB (voice-to-background < 6), -10 dB, or -6 dB (≥ 20) |
| Stationary denoise | ffmpeg present | nr 10 dB if voice-to-background < 6, 8 if < 15 or unknown, 6 if < 25, off above |
| High-pass | Always | 100 Hz if rumble > -12 dB, else 75 Hz |
| Tone match | Always | 50 % toward the speech spectrum, +3 / -6 dB. A PA'd voice is thinner than studio speech, and big low boosts bring up room boom |
| De-ess | Always | Max 6 dB, brightest 20 % of moments |
| Compression and leveler | Always | 2.5:1; slow leveler ±6 dB over 3 s |
| Crowd taming | gap_minus_speech > -8 dB | Gaps capped at speech level -2 dB; every gap frame at least 4 dB down (gentle expansion) and at most 18 dB down. Speech frames move at most 1.5 dB |
| Band extension | Off | A synthetic top on a voice tends to lisp |
| Loudness | Always | -16 LUFS for speech only. In a concert, 4 dB under the music |

## 6. Worked examples

### A. Broadcast concert rip (AAC 160 kb/s, 8:36)

| Measured | Verdict / action |
|---|---|
| -9.6 LUFS, LRA 0.65 LU, PLR 8.8 dB | Heavily limited → de-limit depth 2 |
| Cut-off 15.55 kHz, edge 51 dB | Codec wall → band extension |
| Codec notch 6.4-7.8 kHz, 7.9 dB | Reported, smoothed by the tone match |
| Mid-band negative in 46 % of windows; mono loss -1.6 dB | Anti-phase problem |
| Stem "other": 55 % anti-phase energy, cos -0.99 at 150-1000 Hz, +0.96 above 6 kHz | Repair "other" only (vocal 3.4 % = bleed, left alone) |
| Vocal carries 61 % of 3-10 kHz; 5-8 kHz +6.4 dB vs profile | Strong de-ess |
| Stem shares: bass -12.4 dB vs reference -6.3 dB | Balance: bass +2.8 / +3 dB |
| Stereo-image jump at 211 s | Two sections |
| Tone: 80 Hz -7.7, 315 Hz +3.7, 8 kHz +8.3 dB; RMS deviation 4.09 dB | Tone match |

Results:

| Measure | Before | After |
|---|---|---|
| Tone RMS deviation | 4.09 dB | 1.28 dB |
| 3 s crest | 10.9 dB | 12.3 dB |
| Worst mid-band correlation | -0.49 | -0.12 |
| Mono loss | -1.61 dB | -0.76 dB |
| Loudness | -9.6 LUFS | -10.6 LUFS |

### B. Audience recording of an MC (3:23)

| Measured | Verdict / action |
|---|---|
| -24.6 LUFS, LRA 24.8 LU | Not limited; the crowd jumps |
| Whisper (small); decoding scores ja -0.53, en -1.09, ko -2.33 → Japanese; 386 words; speech 47 % of the time | Talker located by words |
| Gaps +5.6 dB vs speech; loudest 10 % +20.9 dB | Crowd taming |
| Voice-to-background 4.8 dB | Denoise 10 dB; ambience at -12 dB |
| Voice vs speech spectrum: RMS 11.2 dB | Tone match (+3 dB cap) |

Results:

| Measure | Before | After |
|---|---|---|
| Gaps vs speech, median | +5.6 dB | -2.6 dB |
| Gaps vs speech, loudest 10 % | +20.9 dB | -1.4 dB |
| Talker's recognised-word mass (Whisper on the mix) | 281.1 (396 words, mean log-prob -0.51) | 285.9 (391 words, -0.46) |
| Loudness | -24.6 LUFS | -16.1 LUFS |

### C. Synthetic MC where separation fails (test clip)

| Measured | Verdict / action |
|---|---|
| No rhythm section; the vocal stem holds 5 % of the spoken part | Separation lost the voice |
| Whisper word density classifies it as speech | Speech path |
| Talker located on the mix; speech path works on the mix | — |
| Language: decoding scores en -0.31, cy -1.22 | English chosen |

Results:

| Measure | Before | After |
|---|---|---|
| Gaps vs speech | +9.0 dB | -1.7 dB |
| Talker's recognised-word mass (Whisper on the mix) | 41.8 (48 words, mean log-prob -0.17) | 43.8 (49 words, -0.15) |

## 7. When the numbers and the ears disagree

The measurements describe averages against a reference. The listener hears the music.

- If the user says something the numbers do not show, believe the user, find the band or the passage, and adjust.
  `references/tuning.md` maps words to knobs.
- If the numbers say a step is needed but the user likes the original character, turn the strength down. For a
  natural version, see `SKILL.md` step 4.
- A reference from the same artist beats every rule here. If the result sounds "wrong" in a consistent way,
  suspect the profile first.
