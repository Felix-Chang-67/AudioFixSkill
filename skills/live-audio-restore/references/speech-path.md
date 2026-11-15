# Speech path (MC, stage talk): each step in detail

Code: `scripts/audiofix/speech.py`. The talker mask comes from `measure.speech_activity` during `analyze` and is
saved as `speech_mask.npy` (10 ms frames), so restore and verify use the same mask. Keys refer to `plan.json` →
`speech`.

Live talk has different problems from live music:

- The talker reaches the recorder through a PA and a reverberant hall.
- The audience cheers between sentences, often louder than the talk.
- Audience recorders often catch the talk very low (-25 to -45 LUFS).

The goal is to make every word easy to follow without making the room vanish or the voice sound processed.

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

## 3. Stationary denoise (`denoise`)

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

## 4. High-pass (`highpass_hz`)

100 Hz when the voice's energy below 80 Hz is within 12 dB of its 200-800 Hz energy (rumble); otherwise 75 Hz.

## 5. Tone match (`match`)

1. Take the long-term spectrum of the voice over the talker's frames.
2. Compare it with the long-term average speech spectrum (approximately after Byrne et al. 1994, converted to power
   density).
3. Move `alpha` = 50 % toward it, with **boosts capped at +3 dB** and cuts at -6 dB. Nothing is boosted above the
   voice's own bandwidth or below 1.2 × the high-pass.

PA'd voices measure far thinner than studio speech: 12-18 dB less at 125-500 Hz in one MC. Correcting that fully
would bring up the room's boom along with the voice, hence the cap. Typical result: moderate low-mid warmth, a
little presence, sibilance reduced.

## 6. De-ess, compression, leveler

- **De-ess:** as in the music path, at most 6 dB.
- **Compression:** 2.5:1, 5 / 120 ms, threshold at the 60th percentile, make-up gain to the same RMS.
- **Leveler:** a 3 s level of the talker's frames only, ridden toward the median by at most ±6 dB and smoothed over
  1 s. Loud and quiet sentences come closer together.

## 7. Crowd taming (`crowd`)

**Switched on when** the gaps between sentences sit above -8 dB against the speech (median, 400 ms windows).

**How** (10 ms frames, 400 ms level):

1. Outside the talker mask, cap the level at the median speech level minus `cap_db` (2 dB).
2. Lower every gap frame by at least `gap_db` (4 dB, a gentle downward expansion) and by at most `floor_db`
   (18 dB).
3. Release at about 25 dB/s; start reducing up to 120 ms early; smooth over 150 ms.
4. **Speech frames never move by more than 1.5 dB.**

The crowd is held under the talk, not removed: gaps keep their applause at a comfortable level.

**What it cannot do.** Cheering *during* a word stays, because touching it would touch the word.

## 8. Loudness

| File | Target |
|---|---|
| Speech only | -16 LUFS integrated (podcast/web norm), -1 dBTP ceiling |
| Concert with songs | Spoken parts at the music target -4 dB (`master.speech_offset_db`), crossfaded over 1 s at the class boundaries |

## Results on real material

| MC recording | Gaps vs speech | Talker's recognised-word mass (Whisper on the mix) |
|---|---|---|
| 3:23 audience recording | +5.6 → -2.6 dB (loudest 10 %: +20.9 → -1.4 dB) | 281.1 → 285.9; mean log-prob -0.51 → -0.46 |
| One minute of MC inside a song-MC-song file | +1.5 → -3.9 dB (loudest 10 %: +15.9 → -1.0 dB) | 130.7 → 128.6; mean log-prob -0.60 → -0.45 |

The crowd moves a lot and the words stay put: recognition is about the same, with higher confidence. That is the
intended result. The speech path makes the talk easier on the ear without removing anything from the voice.

The verify step re-runs Whisper on input and output (beam search, seeded so a rerun gives the same answer), drops
segments where Whisper loops, and adds up the probabilities of the words inside the talker mask (the
recognised-word mass). Only the talker's words count, because crowd words
going away is the point. A plain word count is too noisy for this check (`references/pitfalls.md`, item 21).
