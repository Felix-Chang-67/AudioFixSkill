# Verification: proving the result without ears

Code: `scripts/audiofix/verify.py`. The `restore` command runs it after rendering; `verify --work W` re-runs it.

The person running this skill, human or model, may not be able to listen critically. So every claim in the
delivery message comes from these numbers, and the user's own listening of `AB.wav` is the final word.

## Checks

| Check | Passes when | What a WARN means |
|---|---|---|
| True peak under the ceiling | ≤ ceiling + 0.1 dB | Should never happen; report it |
| No new clipping | Flat runs/min ≤ max(5, input) | Something overloaded before the limiter |
| Music: closer to the reference tone | RMS deviation 50 Hz-16 kHz (below the cut-off) decreased | Wrong profile for this music, or a very short music part |
| Music: dynamics not squashed further | 3 s crest ≥ input - 0.5 dB | Loudness target too high, or de-limiting off |
| Music: no anti-phase left in the mids | Worst 4 s mid-band correlation ≥ -0.2 | An anti-phase stem was not repaired |
| Music: mono compatible | Mono loss ≥ -1.5 dB | Same, or too much widening |
| Speech: crowd no longer competes with the talk | Gap-vs-speech median ≤ min(input - 3, 0) dB | Talker mask too generous (fallback detector), or cap too loose |
| Speech: the talker's words are recognised at least as well | Whisper on the mix, beam search with a seeded (reproducible) fallback, looping segments dropped: the recognised-word mass inside the talker mask (sum of word probabilities, roughly the number of words recognised correctly) ≥ 90 % of the input's | Denoise or EQ removed speech: back off |
| Speech: closer to the speech spectrum | RMS deviation not up by more than 0.2 dB | Rare; inspect `compare.png` |

Only the checks that apply to the file's content are run.

## Numbers to quote in the delivery

`verification.md` has before and after tables for:

- **Loudness:** LUFS, true peak, PLR, LRA, 3 s crest.
- **Stereo:** side minus mid, worst and mean mid-band correlation, mono loss.
- **Bandwidth.**
- **Music:** deviation from the reference per band.
- **Speech:** gaps against speech (median and loudest 10 %), and the talker's recognised-word mass, word count and
  confidence.

Quote the few that match what the user complained about:

| Complaint | Quote |
|---|---|
| Muddy / harsh / thin | Deviation at the relevant bands |
| Squashed | Crest and PLR |
| Phasey | Correlation and mono loss |
| The crowd drowns the MC | Gap-vs-speech and word counts |

## A/B file

`AB.wav` holds up to four 7 s passages:

- music passages with the most energy;
- spoken passages with the most talk.

Each pair plays the **input, a 0.6 s pause, then the restored version**, with 1.4 s between pairs. Within each pair
the input is matched to the restored version's loudness, so the comparison is not "louder sounds better". The start
times are listed in `verification.md`.

Ask the user to listen on headphones and on a phone speaker. The phone reveals anti-phase and thin bass.

## Listening checklist for the user

Offer it when asking for feedback.

**Songs:**

- Is the low end fuller but not boomy?
- Are the vocals forward and smooth, with no lisp on "s"?
- Do the cymbals shimmer without fizz?
- Do the drums hit?
- Does the stereo image sit around you, not in your head?
- Is there any pumping on loud hits?

**Talk:**

- Is every word easy to follow?
- Does the cheering stay under the voice?
- Does the room still sound like a room?
- Any watery or swirly artefacts on the voice?

**Both:**

- Is anything missing that was there before: a guitar line, ambience?

## Plots

- `overview.png` (from analyze): spectrogram, short-term level and side-minus-mid over time, shaded by content
  class, with dashed lines at the section cuts.
  - A codec wall shows as a flat black ceiling.
  - Anti-phase passages show as side-minus-mid near 0 dB.
  - MC crowds show as level spikes between sentences.
- `compare.png` (from verify): long-term spectra of input, restored and profile, plus before and after spectrograms
  of the A/B passages.
  - The restored curve should follow the profile closely below the input's cut-off.
  - The rebuilt top should continue the profile's slope.
  - Spoken parts should show quieter gaps between words.
