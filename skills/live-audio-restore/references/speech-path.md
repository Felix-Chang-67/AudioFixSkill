# Speech path (MC, stage talk): each step in detail

Code: `scripts/audiofix/speech.py`. The talker mask comes from `measure.speech_activity` during `analyze` and is
saved as `speech_mask.npy` (10 ms frames), so restore and verify use the same mask. Keys refer to `plan.json` →
`speech`.

Live talk has different problems from live music:

- The talker reaches the recorder through a PA and a reverberant hall.
- The audience cheers between sentences, often louder than the talk.
- Audience recorders often catch the talk very low (-25 to -45 LUFS).

The goal is to make every word easy to follow, and the voice sound as good as the restored singing voice, without
making the room vanish.

**Two modes** (`speech.mode`):

| Mode | What the voice gets |
|---|---|
| `v2` (default) | The music path's vocal treatment, the one that made the clearly audible difference on songs: voice EQ toward the reference vocal at the music path's strength (85 %), its vocal compressor (3:1, 4/90 ms), de-limiting when the talk went through the broadcast limiter, band extension above a codec wall |
| `gentle` | The earlier speech path: half-way (50 %) toward the long-term speech spectrum, 2.5:1 compression, no band extension. For listeners who find v2 too processed |

The speech-only steps are the same in both: talker location, room kept underneath, stationary denoise, leveler,
crowd taming.

## 1. Locating the talker (`vad`)

Every later step depends on knowing which 10 ms frames are the talker speaking.

| Method | How | Quality |
|---|---|---|
| `asr` (faster-whisper installed) | Word timings from Whisper. Low-confidence words (p < 0.15) and segments (no-speech > 0.6, or avg log-prob < -1.0) are dropped, known applause hallucinations with them. Spans run from word start -0.12 s to word end +0.25 s; pauses under 0.35 s are closed and blips under 0.1 s dropped | Crowd shouting without words is excluded |
| `harmonic` (fallback) | Autocorrelation peak at 80-400 Hz pitch over 40 ms frames, averaged over 250 ms above 0.35, 125 ms hangover, loud frames only | About F1 0.80 against word timings (precision 0.72, recall 0.88). Part of the crowd is taken for speech, so crowd taming is weaker |

**Which signal is listened to.**

- The vocal stem, if separation kept the voice: `vocal_share` ≥ 0.3 of the spoken parts. Crowd noise mostly goes to
  "other", so the vocal stem gives cleaner timings.
- Otherwise the mix. A robotic, heavily reverberant or crowd-masked voice can end up in "other".

**Language.**

- Give it with `--language` (`ja`, `en`, `zh`, ...) when you know it.
- Otherwise the three likeliest languages from Whisper's language ID each decode a 60 s sample, and the one with the
  best duration-weighted log-probability wins.
- Language ID alone mistook a synthetic English voice for Welsh with 77 % confidence. Decoding scored English -0.31
  against Welsh -1.22.

**Model.** `--asr-model small` is the default: about 480 MB, fast on CPU. Use `medium` or `large-v3` for hard
material if time allows.

## 2. Source (`source`)

- **`stems`.** Voice = vocal stem. Room and crowd = everything else, high-passed at 90 Hz, kept underneath at
  `ambience_db`:

  | Voice-to-background | `ambience_db` |
  |---|---|
  | < 6 dB | -12 |
  | 6 to < 20 dB | -10 |
  | ≥ 20 dB | -6 |

  The room is kept on purpose: talk with no room at all sounds pasted on.
- **`mix`.** When there are no stems, or separation lost the voice (`vocal_share` < 0.3). Everything below works on
  the whole mix.

## 3. De-limit (`delimit`, v2 only)

The De-limiter gain curve at depth 1.5, on the voice and the room, as in the music path.

- **Switched on when** the songs of the same file are de-limited (they show the broadcast limiter, so the talk went
  through the same chain) **and** the talk's peak-to-loudness ratio is under 11 dB.
- **Not on a low PLR alone.** Steady crowd noise or a dense voice also gives a low PLR: a synthetic MC with constant
  crowd noise measured 8.9 dB without any limiter.
- For a broadcast MC on its own (a TV talk segment), switch it on by hand: `speech.delimit.enabled=true`.

## 4. Stationary denoise (`denoise`)

ffmpeg `afftdn`, tracking the noise (`tn=1`). The noise floor is set to the 10th percentile of the voice's 100 ms
levels.

| Voice-to-background | `nr` |
|---|---|
| < 6 dB | 10 dB |
| < 15 dB, or unknown | 8 dB |
| < 25 dB | 6 dB |
| ≥ 25 dB | off |

**Why not a neural denoiser.** On PA'd, reverberant, band-limited live talk, DeepFilterNet3 and Facebook's Denoiser
were tested at full strength. Both removed so much speech that Whisper could hardly transcribe the result. At gentle
settings they did nothing useful. Quality scores without a reference, such as SQUIM, still rated those outputs
highly: they reward "clean", not "the words are still there". Do not use them here.

## 5. High-pass (`highpass_hz`)

100 Hz when the voice's energy below 80 Hz is within 12 dB of its 200-800 Hz energy (rumble); otherwise 75 Hz.

## 6. Voice EQ (`match`)

**Target (v2), first that applies** (`match.target`):

| Target | When | What |
|---|---|---|
| `song_vocal` | The file also has songs, with stems and a profile with stem statistics | The EQ the music path gave the singer's vocal stem (mean over sections, also saved as `vocal_eq.json`) is applied to the talk as it is. Same mic, same PA, same broadcast chain, so the same correction: the talker sounds like the restored singer |
| `studio_vocal` | Talk only, and the music profile has vocal statistics | The studio vocal of the music profile above 400 Hz, the long-term speech spectrum below 250 Hz, blended in between |
| `ltass` | No vocal statistics in the profile | The long-term speech spectrum (approximately after Byrne et al. 1994, converted to power density) |

For a talk-only file from a show whose songs you also have, restore a song first and point the MC at its curve:
`--set speech.match.target=song_vocal speech.match.chain_file=SONGWORK/vocal_eq.json`.

**Why the speech spectrum below 250-400 Hz.** People speak lower than they sing. A studio vocal (singing) is nearly
empty below 200 Hz: -15 dB at 160 Hz and -24 dB at 125 Hz against its 300-1000 Hz level. A speaking voice has its
fundamental and first harmonics there, so matching the talk to the singing vocal down there would thin it.

**How (v2), the music path's per-stem curve:**

1. Measure the voice's long-term spectrum on the talker's frames, normalised at 300-1000 Hz.
2. Move `alpha` = 85 % toward the target. This is the music path's overall strength.
3. Scale boosts by how much the voice dominates the mix in each band. This is full where the voice is at least 30 %
   of the power, and none below 5 %, so the crowd is not boosted along with it.
   - With stems, the voice's share is the vocal stem against the mix.
   - On the mix alone (no stems, or separation lost the voice), the share is the talk minus the quieter half of the
     gaps between sentences, which is the steady background under the talk. On a synthetic MC whose voice stops at
     5.5 kHz, this kept the +3 dB "air" boost off the crowd above it.
4. Cap boosts at +3 dB and cuts at -8 dB. Nothing is boosted below 1.2 × the high-pass or above the voice's own
   bandwidth.

On the 3:23 audience MC (studio-vocal target), the two modes' EQ compare like this:

| Band | v2 | gentle | Effect of v2 |
|---|---|---|---|
| 100 Hz | +3.0 dB | +3.0 dB | Same body |
| 160-200 Hz | -1.6 dB | +1.4 to +1.7 dB | Cuts the PA boom instead of lifting it |
| 1-3 kHz | -2.9 to -5.8 dB | -0.4 to -1.8 dB | Much less "honk" |
| 8 kHz | +3.0 dB | +1.7 dB | More air |
| 12.5 kHz | +3.0 dB | — | More air |

**Gentle mode:** 50 % toward the speech spectrum, boosts capped at +3 dB, cuts at -6 dB.

PA'd voices measure far thinner than studio speech: 12-18 dB less at 125-500 Hz in one MC. Correcting that fully
would bring up the room's boom along with the voice, hence the +3 dB boost cap in both modes.

## 7. De-ess, compression, leveler

- **De-ess:** as in the music path, at most 6 dB.
- **Compression:**
  - v2: the music path's vocal compressor, 3:1, 4 / 90 ms.
  - gentle: 2.5:1, 5 / 120 ms.

  Both set the threshold at the 60th percentile, with make-up gain to the same RMS.
- **Leveler:** a 3 s level of the talker's frames only, ridden toward the median by at most ±6 dB and smoothed over
  1 s. Loud and quiet sentences come closer together.

## 8. Crowd taming (`crowd`)

**Switched on when** the gaps between sentences sit above -8 dB against the speech (median, 400 ms windows).

**How** (10 ms frames, 400 ms level):

1. Outside the talker mask, cap the level at the median speech level minus `cap_db` (2 dB).
2. Lower every gap frame by at least `gap_db` (4 dB, a gentle downward expansion) and by at most `floor_db`
   (18 dB).
3. Release at about 25 dB/s; start reducing up to 120 ms early; smooth over 150 ms.
4. **Speech frames never move by more than 1.5 dB.**

The crowd is held under the talk, not removed: gaps keep their applause at a comfortable level.

**What it cannot do.** Cheering *during* a word stays, because touching it would touch the word.

## 9. Band extension (`band_ext`, v2 only)

**Switched on** for a codec wall below 18.5 kHz (a broadcast rip), so the talk does not sound duller than the songs
around it.

**How:** the same single-sideband shift as in the music path. It is calibrated on the talker's frames, with the band
above the cut-off set 6 dB under the band below the cut-off, then `below_ref_db` (3 dB) lower still: a loud
synthetic top on a voice can lisp.

## 10. Loudness

| File | Target |
|---|---|
| Speech only | -16 LUFS integrated (podcast/web norm), -1 dBTP ceiling |
| Concert with songs | Spoken parts at the music target -4 dB (`master.speech_offset_db`), crossfaded over 1 s at the class boundaries |

## Results on real material

v2 mode:

| MC recording | Target | Gaps vs speech | Talker's recognised-word mass (Whisper on the mix) | Voice vs its target |
|---|---|---|---|---|
| 3:23 audience recording | studio_vocal | +5.6 → -2.5 dB (loudest 10 %: +20.9 → -1.5 dB) | 281.1 → 286.2; mean log-prob -0.52 → -0.49 | 5.8 → 3.4 dB |
| One minute of MC inside a song-MC-song file | song_vocal | +1.5 → -3.9 dB (loudest 10 %: +15.9 → -0.6 dB) | 130.7 → 132.1; mean log-prob -0.60 → -0.50 | — |
| 40 s synthetic MC, separation lost the voice | studio_vocal, on the mix | +9.0 → -1.8 dB | 41.8 → 42.5 | not judged: the voice is the mix |

The crowd moves a lot and the words stay put: recognition is about the same, with higher confidence. The gentle mode
gave the same gaps and word mass within 1-2 %. The modes differ in the voice's tone, which only the listener can
judge: compare them with a three-way A/B (original, gentle, v2).

The song-MC-song file was put together from two different recordings, so its songs and talk do not share a chain.
`song_vocal` is meant for real concert files, where they do. If the talk sounds dull or thin there, switch to
`speech.match.target=studio_vocal`.

The verify step re-runs Whisper on input and output (beam search, seeded so a rerun gives the same answer), drops
segments where Whisper loops, and adds up the probabilities of the words inside the talker mask (the
recognised-word mass). Only the talker's words count, because crowd words
going away is the point. A plain word count is too noisy for this check (`references/pitfalls.md`, item 21).
