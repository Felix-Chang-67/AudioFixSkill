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
| Flat, squashed, no punch, tiring | LRA < 4 LU, or PLR < 11 dB with a 3 s crest < 12.5 dB | Broadcast or streaming limiter | De-limit; drum transients |
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
| PLR (dB) | True peak minus integrated loudness | **< 11 dB with a 3 s crest < 12.5 dB = limited; < 9.5 dB = heavily limited.** Loud modern masters: about 8-9. Open live mixes: 12-16; dense open songs can sit at 10-11 |
| 3 s crest (dB) | Median over 3 s windows of peak / RMS | How much hits stand out locally. Below about 11 dB, punch is gone |

**Rule (music):** limited if LRA < 4, or PLR < 11 together with a 3 s crest < 12.5 dB. If the De-limiter is installed,
de-limit at depth 2.0 when LRA < 2 or PLR < 9.5, otherwise at depth 1.5. Without the De-limiter, drum transients get
stronger shaping instead.

Why the crest condition: a dense but unlimited live song had a PLR of 10.9 dB, a 3 s crest of 13.6 dB and an LRA of
6.7 LU, and the De-limiter found 0.1 dB of limiting in it. PLR alone called it limited, and the limited settings took
its range to 3.1 LU. The limited sources seen so far had crests of 10.3-10.9 dB and LRAs of 0.2-0.7 LU.

### Clipping

The measure counts runs of two or more identical samples within 0.3 dB of the file's peak, per minute.

- **20 or more per minute = digital clipping.** If ffmpeg is available, the mix goes through ffmpeg `adeclip` before
  anything else.
- Lossy codecs smear flat tops, so a broadcast rip of a clipped source can read low. If the user hears crackle on
  loud hits and the count is low, force it with `--set repair.declip.enabled=true --why "..."`.
- A broadcast limiter piles samples up near its ceiling without making flat runs. That is limiting, not clipping.

### Mains hum

The measure takes the peak at 50 or 60 Hz and its first six harmonics, minus the median of the surrounding spectrum.

- **Hum is present if the fundamental stands 10 dB or more above its surroundings, or two harmonics stand 8 dB or
  more above theirs.**
- The fix is notch filters (Q 35) on every harmonic at 6 dB or more.
- Small excesses (2-6 dB) are normal. Bass notes land on those frequencies too.

### Whistles in the highs (codec tones, steady or on and off)

Two measures look at the highs, 4-20.8 kHz, on the signal the render uses (after codec repair).

- **Steady lines.** A narrow line stands 8 dB or more over its +-160 Hz neighbourhood in the long-term spectrum. It
  counts if it also stands 6 dB over its neighbours in at least 45 % of the frames.
  - The floor was 60 % below 12 kHz until a 6.27 kHz line, 8 dB over its neighbours in 53 % of a song's frames, was
    left in. At 45 %, the other eight recordings the skill was tuned on showed nothing below 12 kHz.
- **On-and-off lines.** In 93 ms frames, a narrow peak 12 dB or more over its +-150 Hz neighbourhood is one whistle
  if it holds its frequency (within 11 Hz) for 0.5 s or longer.
  - Whistles within 12 Hz of each other make one line. A line counts when its whistles add up to 2 s or more, with
    a median excess of 18 dB or more.
  - At most 48 lines, the longest and strongest first. The frequency is read to a fraction of a bin: a 15 Hz notch
    5 Hz off a line cuts it by only 5 dB.
- **Why it matters.** Music holds no tone at a fixed frequency in the top octaves for most of a song, and it does
  not come back to one exact frequency, 18 dB over its neighbours, for seconds in total. A low-bitrate encoder does
  both.
  - HE-AAC's band replication switches sinusoids on and off at the centres of its 344.5 Hz bands (at 44.1 kHz:
    6.72, 7.06, 7.75, 8.44 ... kHz).
- **Example.** A 50 kb/s HE-AAC rip carried four lines at 8.44, 9.13, 9.82 and 10.51 kHz, 689 Hz apart and 14-19 dB
  over their neighbours, through the whole song.
  - The restoration's lift of the highs raised them by about 10 dB.
  - The band extension copied the 10.51 kHz one up to 13.2, 16.6 and 19.1 kHz.
  - The listener heard a "digital feel".
  - With those four notched, the listener still heard whistles in every version. 7.75 kHz whistled for 20 s in
    total, 6.72 kHz for 6 s and 7.06 kHz for 3 s, 23-32 dB over their neighbours while on. They were in under 60 %
    of the frames, the steady measure's floor then, so it missed them.
- **Codec repair adds its own.** Apollo's output carried 3-5 steady lines at 17-20.1 kHz on every song it processed.
  It also carried 8-40 on-and-off lines at multiples of 100 Hz, from 12.7 kHz up.
- **No false alarms found.** On the seven other recordings the skill was tuned on, the on-and-off measure found
  nothing below 12.7 kHz. Above that, it found only the codec repair's lines.
- **The fix.** A 15 Hz-wide notch on each line (`repair.birdies`), on the mix and on every stem, before anything is
  boosted or copied up. The four steady lines took 1.3 dB of the 8-11 kHz band with them.
- **Still heard?** `investigate` measures both kinds in the render, and its variant notches whatever is left.

### Notes copied up into the highs (band replication)

A low-bitrate HE-AAC encoder keeps only the low band and rebuilds the top from a copy of it, shifted up by a whole
number of its bands (344.5 Hz at 44.1 kHz, 375 Hz at 48 kHz). Every held note below reappears up there that many Hz
higher, out of tune with its own harmonics: a short whistle or a metallic ring on each note.

- **The measure.** The tonal fine structure (dB over the +-150 Hz median) at f is correlated with that at f - D, for
  every shift D from 2 to 9 kHz, with f from 4 kHz up to the codec's cut-off (the replication stops there).
  - A shift counts at a correlation of 0.2 or more. Its band is the run of 500 Hz bands around the best one that
    correlate 0.15 or more.
- **Calibration.**
  - A 50 kb/s rip scored 0.43 at 5512.5 Hz (16 bands): 1-5.9 kHz was copied into 6.5-11.4 kHz. Codec repair left
    the copies in (0.36 after it).
  - A broadcast scored 0.27 at 6750 Hz (18 bands of 375 Hz, a 48 kHz source), in 7.5-14 kHz.
  - Six recordings without band replication found nothing over 0.1 below their cut-off.
  - The codec repair's own 100 Hz comb, above the cut-off, scored 0.10-0.14 at multiples of 100 Hz. The cut-off bound
    and the 0.2 floor keep it out.
- **Example.** Once the steady and on-and-off whistles of that rip were notched, 21 short whistles were left. Three
  of the strongest were notes from the low band, 5512.5 Hz higher: 1174 Hz at 6686 Hz, 1658 Hz at 7171 Hz, 1324 Hz at
  6837 Hz. The restoration had made the one at 6686 Hz 13 dB more prominent.
- **The fix** (`repair.sbr_copies`). In each 93 ms frame, a peak in the band that stands 6 dB or more over its
  +-150 Hz median is cut to that median (at most 30 dB), when its source, the shift below, peaks there too.
  - The cut is measured on the mix and applied alike to the mix and every stem.
  - On that rip, 21 % of the band's bins were cut by 8 dB on average. The band lost about 3 dB, and the shift's
    correlation fell to nothing. The tone match puts the band back toward the reference.

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
- **Without stems there is no content map.** The rhythm section is measured on the drum and bass stems, and Whisper
  alone cannot tell talk from singing, because lyrics are words too. The whole file is treated as music, with a
  warning in `diagnosis.md`. For an MC, pass `--content speech`.

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

- Whenever the user says what the file is: `--content music` for a song, `--content speech` for an MC or talk.
  The automatic split is for files that hold both.
- A file that is all MC or all music but was split anyway: use `--content speech` or `--content music`.
- Quiet sung passages without drums are the weak spot. Singing pauses between phrases (pause share 0.36-0.42 on
  one such song, above the 0.25 rule), and Whisper writes sung lyrics down as words (about 1 word/s). Neither rule
  tells singing from talk there.
- A known feed change the detector missed: add it to `sections` in `plan.json`.

## 4. Music path: decisions

`diagnosis.md` has a stem table (level against the mix, anti-phase energy, worst and median correlation, anti-phase
phase by band), the 3-10 kHz share by stem, and the tone deviation against the music profile by band.

| Step | Switched on when | Strength / parameters |
|---|---|---|
| Codec repair (Apollo, during analyze) | Cut-off below 16 kHz and the band under it sparse (8-12.5 kHz spectral flatness < 0.15), Apollo installed (`--codec-repair auto`) | Refills the sparse, peaky band under the cut-off (8-12.5 kHz flatness 0.08 → 0.26 on an audience rip; studio 0.42) before anything else. About 6x real time on CPU |
| De-limit | Limited (LRA < 4, or PLR < 11 with a 3 s crest < 12.5), and the De-limiter is installed | Depth 2.0 if LRA < 2 or PLR < 9.5, else 1.5 |
| Drum transients | Stems available | 0.35 / max 2.5 dB with de-limiting; 0.55 / 4.5 dB if limited but no de-limiter; 0.25 / 2 dB otherwise |
| Polarity repair | A stem has 10 % or more of its energy in 4 s windows with L-R correlation below -0.3 (a whole layer reversed), **or** 5 % or more of its energy in coherent anti-phase bins together with a mono loss beyond -1 dB (anti-phase in some bands only) | Only that stem. Without stems, the mix when 10 % or more of mid-band windows are negative |
| De-ess (vocal) | Vocal stem present | Max cut 9 dB if the vocal carries 50 % or more of the 3-10 kHz energy, or 5-8 kHz sits more than 4 dB above the profile; else 6 dB. Acts only on the brightest 20 % of moments |
| Widen guitars/keys | Stereo source, stems, and the drum stem's median L-R correlation ≥ 0.8 (a mixed image) | Side +3 dB above 150 Hz. Never on mono or near-mono. Off on a diffuse, room-miked recording (drums below 0.8): it is already wide. An audience recording measured drums 0.65, vocal 0.32, side only 3.8 dB under mid; broadcast mixes measured 0.98-1.0 |
| Per-stem tone match | Stems and a profile with stem statistics | 60 % toward the reference stem. Boosts at most +3 dB (+4 dB below 150 Hz; the vocal's 1-8 kHz presence +6 dB), cuts at most -6 dB. Boost only where the stem carries at least 5-30 % of the band (no lifting of bleed). Bass and kick never cut below 120 Hz |
| Width per stem | Room-miked image (drums' median L-R correlation < 0.8) and stems | Vocals, drums and bass narrowed toward the reference stems' side level (studio: vocals -31, drums -23, bass -57 dB re mid), at most 20 dB; guitars keep their width |
| Stem balance | Same, with drums in the performance | Each stem toward its reference share of the mix, per section: 50 % and at most ±3 dB on mixed images, 75 % and at most ±6 dB on room-miked ones. A stem below -30 dB re mix is absent and left alone. No drums (below -30 dB) means an acoustic or stripped-down arrangement: the full band's ratios do not apply, and the step is off |
| Stem dynamics | Stems | Limited source: drum peaks held at drum loudness +11 dB (1 ms lookahead); bass 3:1, vocal 3:1, guitars 2.5:1; bus 1.8:1. Open source: drums at +15 dB; bass 2:1, vocal 2:1, guitars 1.5:1; no bus glue |
| Loudness contour | Open source (LRA ≥ 4 and PLR ≥ 11) | After all processing, the original's 3 s loudness contour is put back, as a slow gain within ±6 dB |
| Overall tone match | Always (with a profile) | 85 % toward the profile, -8 to +6 dB, per section. No boosts above the cut-off, nor (stem matches too) above the start of the top-band fill, which sets the level there |
| Band extension | Brick wall below 18.5 kHz, or codec damage as above | Starts 500 Hz under the lowest cut-off the codec used (the 10th percentile of 15 s windows; it moves over time). An SBR-style shifted copy of the band below, half of it noise-like (`noise_mix` 0.5). Per 20 s section and 250 Hz band, only what is missing up to the reference's own level is added, so the top follows the studio version's roll-off and never adds to content that is already there |
| Bass to mono | Stereo | Below 100 Hz |
| Loudness | Always | Loudest target where the limiter exceeds 1 dB at most 8 % of the time and 2 dB at most 1.5 % (limited source), or 2.5 % and 0.5 % (open source). Open sources also keep their 3 s crest within 0.5 dB of the original: the target steps down 0.5 dB until it does, from -8 LUFS down to the original's loudness minus 0.5 dB (at least -14, at most -20). The crest asked for is never more than the music brings to the master |

**Limited or open: the decision behind four steps.** LRA < 4 LU, or PLR < 11 dB with a 3 s crest under 12.5 dB, means a limiter already flattened
the recording (broadcasts, loud uploads). Undoing that is part of the restoration, and the master may be loud.

Otherwise the recording is open (audience and board recordings, careful rips), and its dynamics are part of the
performance. On one open audience recording (LRA 10 LU), the limited-source settings took the range down to 6.4 LU
and raised the quiet intro by 4-5 dB against the choruses. Tone and stem changes alone, with no compression at all,
still took 2 LU off. Hence light compression plus the contour step for open recordings.

**Coherent anti-phase energy by stem.** The repair's own detector, measured on all test material:

| Stem | Coherent anti-phase energy | Mono loss | Verdict |
|---|---|---|---|
| Drums and bass, every recording | 0-0.1 % | 0 to -1 dB | Leave alone |
| Vocal stems (broadcast, audience) | 6.8-8.9 % | -1.2 to -2.0 dB | Repair: mono loss improved by 0.4-0.5 dB |
| Guitar stem, audience recording | 6.0 % | -1.7 dB | Repair (the window rule saw 0 %) |
| Whole reversed guitar layers | 55-76 % | -3.7 to -6.1 dB | Repair (the window rule saw 55-100 %) |

**Reading the anti-phase picture.** In the "anti-phase by band" column, cosines near -1 in 150-2500 Hz with +1 above
6 kHz mean a coherent layer reversed only in its body. A plain channel swap would be reversed everywhere. The repair
rotates only coherent anti-phase bins, so this pattern is handled correctly.

In one broadcast, the vocal stem showed 3.4 % of its energy in anti-phase windows, below the window threshold. The
detector found 8.7 % coherent anti-phase energy in it, mostly bleed of the reversed guitar layer, with a mono loss of
-1.4 dB. It is now repaired too, which brings that loss to -0.9 dB.

**Reading the tone table.** "+" means too much against the reference. A typical broadcast or audience pattern:

| Band | Deviation | Meaning |
|---|---|---|
| 80 Hz | -6 to -9 dB | No weight |
| 315 Hz | +4 to +6 dB | Mud |
| 5-8 kHz | +5 to +10 dB | Harshness |

Check the 3-10 kHz share before blaming cymbals. On the broadcast example, the vocal carried 61-68 % of it.

**Choosing the profile matters more than any knob.** Genre sets the target. A jazz club recording matched to a loud
rock profile will be pushed toward rock. Build a profile from the artist or the genre whenever possible.

Best of all is the studio version of the very song. `analyze` and `auto` look it up themselves when no
`--music-profile` is given:

- **Artist and title:** from the file's tags, or `--artist` / `--title`. Track numbers and live markers are removed,
  and titles such as MC or SE are skipped.
- **Stores:** tried in an order set by the script of the names. Kana → jp, us; hangul → kr, us; ideographs → jp, tw,
  hk, cn, us; otherwise us, gb, jp. `--country` puts one first.
- **Fallback:** offline, untagged or not found means the bundled profile, with a `[next]` line asking for the studio
  track: `profile --files TRACK --song`. By hand: `profile --itunes-artist ARTIST --itunes-song TITLE`.

On an audience recording, the artist's 27-song average left 100-400 Hz 5-12 dB under that song's own studio
master. Songs differ: that one has a loud bass. Its own 30 s preview brought the section within ±3 dB.

**Like with like.** A preview is one passage, usually the chorus, and a live recording's tone can change a lot
between passages: one measured 8 dB more low end in the previewed passage than in the song as a whole. Matching
the whole song's average to the preview over-boosted the lows by 5-7 dB in that very passage.

A same-song profile therefore carries the preview's chroma, and `analyze` finds the passage in the recording. Tone
and stem balance are measured there, and the resulting fixed EQ applies to the whole song. On the test song this
brought the mix within 1.7 dB RMS of the studio master in that passage, against 2.9 dB with the whole-song average.

**Finding the passage.**

- Each chroma frame has its mean removed, so a flat frame (silence, noise, a dense distorted wall) scores 0.
- The score is the mean per-frame correlation, searched over tempo 0.94-1.06 and key shifts of -2 to +2 semitones.
- It must reach 0.35 and beat the best other key shift by 0.1. Another song scores about the same in every key.
- Calibration on four live recordings and two previews: the same song scored 0.46 and 0.67, other songs at most
  0.24. The raw cosine used before scored 0.82-0.91 for every pair, silence included, and accepted a wrong song at
  0.91.
- A looked-up preview whose passage is not found is dropped for the bundled profile. A wrong song or another
  arrangement is a worse target than the generic one.

`--itunes-song` keeps the studio version: an exact title match without "live", "remix", "instrumental" and
the like. The first try picked up a live-album version too.

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

The voice is treated like the vocal stem of the music path (`speech.mode` v2, the default). `gentle` keeps the
earlier, lighter treatment; the table gives both where they differ.

| Step | Switched on when | Strength / parameters |
|---|---|---|
| Source | Stems available and vocal_share ≥ 0.3 | Voice from the vocal stem; the rest high-passed at 90 Hz underneath at -12 dB (voice-to-background < 6), -10 dB, or -6 dB (≥ 20) |
| De-limit (v2) | The songs of the same file are de-limited and the talk's PLR < 11 dB | Depth 1.5. A low PLR alone does not count: steady crowd noise gives one too. Broadcast MC on its own: switch on by hand |
| Stationary denoise | ffmpeg present | nr 10 dB if voice-to-background < 6, 8 if < 15 or unknown, 6 if < 25, off above |
| High-pass | Always | 100 Hz if rumble > -12 dB, else 75 Hz |
| Voice EQ, target (v2) | First that applies | `song_vocal`: the file has songs with stems, so their vocal-stem EQ is copied (same chain). `studio_vocal`: the profile's studio vocal above 400 Hz, the speech spectrum below 250 Hz. `ltass`: the speech spectrum |
| Voice EQ, strength | Always | v2: 85 % toward the target, boosts only where the voice dominates the mix, +3 / -8 dB. gentle: 50 % toward the speech spectrum, +3 / -6 dB. Boosts stay at +3 dB in both: a PA'd voice is thinner than studio speech, and big low boosts bring up room boom |
| De-ess | Always | Max 6 dB, brightest 20 % of moments |
| Compression and leveler | Always | v2: 3:1, 4/90 ms (the music path's vocal compressor). gentle: 2.5:1. Slow leveler ±6 dB over 3 s |
| Crowd taming | gap_minus_speech > -8 dB | Gaps capped at speech level -2 dB; every gap frame at least 4 dB down (gentle expansion) and at most 18 dB down. Speech frames move at most 1.5 dB |
| Band extension (v2) | Codec wall below 18.5 kHz | As for the songs, 3 dB gentler (a loud synthetic top on a voice can lisp). Off in gentle mode |
| Loudness | Always | -16 LUFS for speech only. In a concert, 4 dB under the music |

## 6. Worked examples

### A. Broadcast concert rip (AAC 160 kb/s, 8:36)

| Measured | Verdict / action |
|---|---|
| -9.6 LUFS, LRA 0.65 LU, PLR 8.8 dB | Heavily limited → de-limit depth 2 |
| Cut-off 15.55 kHz, edge 51 dB | Codec wall → band extension |
| Codec notch 6.4-7.8 kHz, 7.9 dB | Reported, smoothed by the tone match |
| Mid-band negative in 46 % of windows; mono loss -1.6 dB | Anti-phase problem |
| Stem "other": 55 % anti-phase energy, cos -0.99 at 150-1000 Hz, +0.96 above 6 kHz | Repair "other". The vocal stem has 3.4 % in anti-phase windows but 8.7 % coherent anti-phase energy (bleed of the reversed layer) and -1.4 dB mono loss: repaired too |
| Vocal carries 61 % of 3-10 kHz; 5-8 kHz +6.4 dB vs profile | Strong de-ess |
| Stem shares: bass -12.4 dB vs reference -6.3 dB | Balance: bass +2.8 / +3 dB |
| Stereo-image jump at 211 s | Two sections |
| Tone: 80 Hz -7.7, 315 Hz +3.7, 8 kHz +8.3 dB; RMS deviation 4.09 dB | Tone match |

Results:

| Measure | Before | After |
|---|---|---|
| Tone RMS deviation | 4.09 dB | 1.28 dB |
| 3 s crest | 10.9 dB | 12.3 dB |
| Worst mid-band correlation | -0.49 | +0.09 |
| Mono loss | -1.61 dB | -0.66 dB |
| Loudness | -9.6 LUFS | -10.6 LUFS |

### B. Audience recording of an MC (3:23)

| Measured | Verdict / action |
|---|---|
| -24.6 LUFS, LRA 24.8 LU | Not limited; the crowd jumps |
| Whisper (small); decoding scores ja -0.53, en -1.09, ko -2.33 → Japanese; 386 words; speech 47 % of the time | Talker located by words |
| Gaps +5.6 dB vs speech; loudest 10 % +20.9 dB | Crowd taming |
| Voice-to-background 4.8 dB | Denoise 10 dB; ambience at -12 dB |
| Talk only; music profile has vocal statistics | Voice EQ target `studio_vocal` |
| Voice: PA boom at 160-200 Hz, "honk" at 1-3 kHz, little air above 8 kHz against the target; 5.8 dB RMS off | Voice EQ 85 %: -1.6 dB at 160 Hz, -3 to -6 dB at 1-3 kHz, +3 dB at 8-12.5 kHz, +3 dB at 100 Hz |

Results (v2 mode):

| Measure | Before | After |
|---|---|---|
| Gaps vs speech, median | +5.6 dB | -2.5 dB |
| Gaps vs speech, loudest 10 % | +20.9 dB | -1.5 dB |
| Talker's recognised-word mass (Whisper on the mix) | 281.1 (396 words, mean log-prob -0.52) | 286.2 (397 words, -0.49) |
| Voice vs its target, talker's frames | 5.8 dB RMS | 3.4 dB RMS |
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
| Gaps vs speech | +9.0 dB | -1.8 dB |
| Talker's recognised-word mass (Whisper on the mix) | 41.8 (48 words, mean log-prob -0.17) | 42.5 (48 words, -0.16) |
| Voice EQ boosts above the voice's 5.5 kHz band edge | — | none: the voice's share there, talk minus the quieter gaps, is nil |

### D. Open audience recording of a song (5:33)

| Measured | Verdict / action |
|---|---|
| -15.4 LUFS, LRA 10.0 LU, PLR 15.5 dB | Open: no de-limiting; light dynamics, contour step, gentle master |
| Cut-off 13.1 kHz, edge 47 dB | Codec wall → band extension |
| Tone: 200-800 Hz +2 to +6 dB, 2-10 kHz -4 to -10 dB; RMS 5.1 dB | Tone match |
| Bass stem -17.4 dB against the mix (reference -6.3) | Balance: bass +3 dB |
| Coherent anti-phase energy: vocal 8.9 %, guitars 6.0 %; their mono loss -2.0 / -1.7 dB (window test: 3.7 % / 0 %) | Polarity repair on both |
| Drums' median L-R correlation 0.65; side 3.8 dB under mid | Room-miked image: no widening |

Results with the defaults, no hand edits:

| Measure | Before | After |
|---|---|---|
| Tone RMS deviation | 5.08 dB | 1.89 dB |
| Loudness range | 10.0 LU | 9.2 LU |
| 3 s crest | 14.8 dB | 15.0 dB |
| Loudness | -15.4 LUFS | -13.5 LUFS |
| Mono loss | -1.62 dB | -1.24 dB |
| Bandwidth | 13.1 kHz | rebuilt to about 20 kHz |

Before these rules existed, two assistants hand-tuned this song:

| Version | Tone deviation | Loudness range | Loudness |
|---|---|---|---|
| Claude, limited-source dynamics | 1.80 dB | 6.4 LU | -13.0 LUFS |
| ChatGPT, own edits | 2.26 dB | 9.0 LU | -16.0 LUFS, quieter than the source |

The defaults now keep the better half of each.

## 7. When the numbers and the ears disagree

The measurements describe averages against a reference. The listener hears the music.

- If the user says something the numbers do not show, believe the user, find the band or the passage, and adjust.
  `references/tuning.md` maps words to knobs.
- If the numbers say a step is needed but the user likes the original character, turn the strength down. For a
  natural version, see `SKILL.md` step 4.
- A reference from the same artist beats every rule here. If the result sounds "wrong" in a consistent way,
  suspect the profile first.
