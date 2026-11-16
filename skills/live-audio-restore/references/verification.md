# Verification: proving the result without ears

Code: `scripts/audiofix/verify.py`. The `restore` command runs it after rendering; `verify --work W` re-runs it.

The person running this skill, human or model, may not be able to listen critically. So every claim in the
delivery message comes from these numbers, and the user's own listening of `AB.wav` is the final word.

## Checks

| Check | Passes when | What a WARN means |
|---|---|---|
| True peak under the ceiling | ≤ ceiling + 0.1 dB | Should never happen; report it |
| No new clipping | Flat runs/min ≤ max(5, input) | Something overloaded before the limiter |
| Music: closer to the reference tone | RMS deviation 50 Hz-16 kHz (below the cut-off) decreased. With a same-song reference, measured on the passage it shows | Wrong profile for this music, or a very short music part |
| Music: dynamics not squashed further | 3 s crest ≥ input - 0.5 dB | Loudness target too high, or de-limiting off |
| Music: the song's quiet and loud parts kept apart (open recordings only) | Loudness range (LRA) ≥ input - 2 LU | The song was flattened: compression too firm, or the contour step off |
| Music: not quieter than the original (open recordings only) | Integrated loudness ≥ input - 0.5 dB | Master criteria set too strict: a quieter result reads as "nothing changed" |
| Music: no anti-phase left in the mids | Worst 4 s mid-band correlation ≥ -0.2 | An anti-phase stem was not repaired |
| Music: mono compatible | Mono loss ≥ -1.5 dB | Same, or too much widening |
| Speech: crowd no longer competes with the talk | Gap-vs-speech median ≤ min(input - 3, 0) dB | Talker mask too generous (fallback detector), or cap too loose |
| Speech: the talker's words are recognised at least as well | Whisper on the mix, beam search with a seeded (reproducible) fallback, looping segments dropped: the recognised-word mass inside the talker mask (sum of word probabilities, roughly the number of words recognised correctly) ≥ 90 % of the input's | Denoise or EQ removed speech: back off |
| Speech: voice closer to its target | On the talker's frames, the RMS deviation from the target the render aimed at is not up by more than 0.2 dB. The target is the studio vocal above 300 Hz with the speech spectrum below, or the speech spectrum in gentle mode | Wrong music profile for this talker (its vocal statistics set the target), or denoise reshaping the voice |

Only the checks that apply to the file's content are run. The voice-tone check is skipped in two cases; its numbers
stay in `verification.json`:

- **The songs' vocal EQ was copied** (`song_vocal`): there is no spectral target to measure against.
- **The voice is the mix** (no stems, or separation lost the voice): the background owns part of the spectrum, and
  removing it moves the measured "voice" too.

## Review: what PASS does not cover (`review.md`)

`verify` also writes `review.md`, and prints each item as a `[review]` line. Every item needs an answer in the
delivery.

| Item | From | What to tell the listener |
|---|---|---|
| applause | Timeline windows with no vocal, the drum stem carrying them, and peaks 20 dB or more over their level | It was processed with the music and its claps held back; ask whether to keep, soften or cut it |
| arrangement | Drums or bass below -30 dB re mix outside applause | Stripped-down performance, or a bass separation could not find: what the reference's low end means here |
| peaks | PLR above 18 dB, with the largest peaks' time, width and brightness | Narrow, bright peaks are claps or knocks near the recorder; holding them back is right |
| reference | Same-song match under 0.55, or the generic profile | How far "like the studio version" can go |
| highs | Codec repair applied | The top octave is rebuilt, and how to make it quieter |
| warn | A failed check, traced through the step ledger | Which step moved the measure, by how much, and the knob that was changed |
| limits | Overall EQ at its +6 dB cap, or a stem balance move at its limit | What stayed short and why a bigger boost would only lift noise |
| edits | The plan used against the default plan | Every edit, with its reason |
| previous | The plan and checks of the render before | What changed, and the comparison file |

**The step ledger (`steps.json`).** After every music step, the render records the music's loudness, 3 s crest,
loudness range, mono-sum loss and tone deviation from the reference. A WARN is traced to the steps that moved its
measure the wrong way.

Example: on a vocal-led acoustic song, the crest lost 0.5 dB in the stem dynamics (the vocal compressor). The drum
stem's peak limit was needed: it held back the claps of the applause. Without the ledger, the cause took four
control renders to find.

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
| Flat, no build-up (open recordings) | LRA |
| Phasey | Correlation and mono loss |
| The crowd drowns the MC | Gap-vs-speech and word counts |

## A/B file

`AB.wav` holds up to four 7 s passages:

- the music passages with the most energy, plus the quietest passage that is still music, which reveals hiss,
  pumping and raised quiet parts;
- the spoken passages with the most talk.

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
- `compare.png` (from verify): long-term spectra of input, restored and profile; before and after spectrograms of
  the A/B passages; and a last row with the short-term loudness over the whole file and its spread around the
  median.
  - The restored curve should follow the profile closely below the input's cut-off.
  - The rebuilt top should continue the profile's slope.
  - Spoken parts should show quieter gaps between words.
  - On an open recording, the restored loudness curve should keep the input's shape. Quiet intros and breaks should
    stay as far below the choruses as before, and the spread should be about as wide.
