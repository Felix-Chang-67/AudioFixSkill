# Music path: each step in detail

Code: `scripts/audiofix/music.py` (steps 1-10). The repairs on the raw mix live in `pipeline.py`; loudness lives in
`master.py`. Keys refer to `plan.json` → `music`.

The music path is the "strong remaster". On a limited broadcast it was validated against a hand-made version of the
same pipeline: the long-term spectra of the two agreed within 0.13 dB RMS.

- **Tone:** it aims for the artist's studio masters, not just a cleaner copy of the broadcast.
- **Dynamics:** it depends on the source. A limited one is opened up and mastered loud. An open one keeps its own
  dynamics (steps 6 and 10, and the master).

## 0a. Codec repair (`repair.codec`, Apollo; runs during `analyze`)

**Why.** A low-bitrate AAC or MP3 does more than cut the top. Below its cut-off it zeroes whole time-frequency tiles,
so the remaining band is sparse and peaky: "birdies" and holes. Two later steps make that audible:

- The tone match lifts this band toward the reference.
- The band extension copies it upward.

At the right level but with the wrong texture, the highs sound fake and harsh. On an audience recording, the 8-12.5
kHz band had a spectral flatness of 0.08, against 0.42 for the studio master of the same song (1 = noise-like, 0 =
sparse tones).

**How.** Apollo (Li et al., a network trained to restore lossy-coded music) refills that texture before
separation:

| Band | Before | After Apollo | Studio master |
|---|---|---|---|
| 8-12.5 kHz | 0.08 | 0.26 | 0.42 |
| 4-8 kHz | 0.19 | 0.28 | 0.45 |

The tone is otherwise unchanged. Its own extension above the cut-off is weak (27 dB under the studio level), so the
band extension (step 8) still runs. It now works from a natural band.

**When.** `--codec-repair auto` (the default) runs it when the cut-off is below 16 kHz and the band under it is
sparse: an 8-12.5 kHz spectral flatness below 0.15. Apollo must be installed.

| Band texture (flatness) | Value |
|---|---|
| Damaged rips | 0.06-0.07 |
| Healthy bands: a 160 kb/s broadcast, an AAC 128 kb/s file, a studio master | 0.28-0.41 |

The shape of the cut-off does not decide. MP3-style encoders roll off over a 1-1.5 kHz transition band instead
of a brick wall: one rip fell 18 dB between 12.7 and 14.3 kHz and showed only a 7.7 dB "edge", which a
wall-only rule missed. Everything after it uses `input_repaired.wav`. The cut-off and the global numbers stay those of the
original, and `verify` compares with the original.

**Cost.** On the CPU, about 6x real time: a 5 min song takes about 30 min.

- It runs on an NVIDIA (CUDA) or Apple silicon (MPS) GPU when one is found (`device.py`), which is much faster.
- A failing GPU run (out of memory, an operation MPS lacks) falls back to the CPU.
- `AUDIOFIX_DEVICE=cpu|cuda|mps` forces a device.
- Without a GPU, use `--codec-repair off` for long concerts, or run it overnight.

**Safety.** The checkpoint is loaded with torch's restricted unpickler and an allowlist of the plain classes it
references. The omegaconf classes are stand-ins, so nothing in the file can run code.

## 0b. Separation

The input goes through demucs `htdemucs` into four stems: vocals, drums, bass and other (guitars, keys, crowd).

- The **residual**, the mix minus the sum of the stems (about -28 dB), is carried along untouched. So "do nothing"
  reproduces the input exactly.
- Each per-stem step works on stems; the residual only receives the de-limiting gain.
- Separation errors are bleed between stems. The pipeline keeps per-stem changes moderate (±3 dB balance, capped
  EQ) so bleed never becomes an audible artefact.

## 1. De-limit (`delimit`)

**Why.** A broadcast or streaming limiter turns every hit down, so the music loses its punch and becomes tiring. You
cannot simply expand it: you would need to know what the limiter did.

**How.**

1. The De-limiter network (Jeon & Lee, WASPAA 2023) predicts the limiter's gain reduction for every sample.
2. Its curve, relative to the median, is applied with `depth` times its swing (2.0 = twice as deep) to every stem and
   the residual. The limiter acted on the whole mix, so one curve fits all stems.
3. The model runs in 20 s chunks with 2 s crossfades on CPU (516 s took 112 s). The curve is cached in
   `delimit_gain.npy`.

**Parameters.** `depth` 1.5-2.0. On real broadcast material the predicted swing was only about 1.6 dB between the
1st and 99th percentiles (the model was trained on a different limiter), hence the doubling.

**Failure mode.** Too deep, and quiet passages pump up and down. Lower `depth`.

## 2. Polarity repair (`antiphase.stems`)

**Why.** A layer recorded or mixed in anti-phase (L = s, R = -s):

- sounds diffuse and "outside the head" on headphones;
- loses body on speakers;
- cancels in mono: phones, Bluetooth speakers, club systems.

**How (`dsp.fix_antiphase`).**

1. Take the STFT of L and R (4096-sample window, 1024 hop).
2. In every bin, average the cross-spectrum over about 1 s and ±2 bins.
3. Compute two quantities:
   - coherence: is it the same signal in both channels?
   - the cosine of the phase: is it in phase or reversed?
4. `anti = clip((-cos - 0.15)/0.55) × clip((coh - 0.12)/0.33)`, smoothed over about 0.5 s.
5. Rotate R by `π·anti`. This is a unit-modulus rotation, so the level is unchanged and nothing is cancelled in the
   transitions.
6. Long files are processed in 60 s blocks with 4 s crossfades.

**Effect on a broadcast guitar layer.**

| Measure | Before | After |
|---|---|---|
| Stem L-R correlation | -0.68 | +0.74 |
| 6-15 kHz (already in phase) | +0.97 | +0.97, untouched |
| Level of R | — | changed by 0.05 dB |
| That layer in mono | — | 7.4 dB louder: no longer self-cancelling |

The result is still wide (correlation 0.74), not mono.

**Which stems.** A stem is repaired when either test holds:

- **A whole reversed layer:** 10 % or more of its energy sits in 4 s windows with L-R correlation below -0.3.
- **Anti-phase in some bands only:** the repair's own detector finds 5 % or more of its energy in coherent anti-phase
  bins, and the stem loses more than 1 dB in mono. The detector runs on up to 120 s spread over the stem.

  On an audience recording, the guitar stem was reversed only below 2.5 kHz, and the vocal stem only at 150-400 Hz.
  The window test saw 0 % and 3.7 %; the detector saw 6.0 % and 8.9 %. Repairing both took the stems' mono loss from
  -1.74 and -2.04 dB to -1.44 and -1.55 dB.

Drum and bass stems measured 0-0.1 % on every recording. Because only coherent anti-phase bins are rotated, a
repaired stem with little anti-phase barely changes: everything in phase or merely decorrelated is left as it was.

## 2b. Width per stem (`width_match`, room-miked images)

**Why.** In a studio master the vocal, kick, snare and bass sit in the middle; only guitars and keys are spread.
Here is the side level re mid of each stem in the studio master of one song, and in an audience recording of it:

| Stem | Studio master | Audience recording |
|---|---|---|
| Vocals | -31 dB | -3 dB |
| Drums | -23 dB | -7.5 dB |
| Bass | -57 dB | -16 dB |
| Guitars/keys ("other") | +0.2 dB | -3 dB |

The extra width of the vocal and drums is the room: reflections that reach the two mics differently. It reads as
diffuse, unfocused and impure, and it costs mono compatibility.

**How.** Only on room-miked images (drum stem's median L-R correlation below 0.8):

1. Each of vocals, drums and bass has its side signal scaled down toward the reference stem's side level. The target
   comes from the profile if it holds widths, otherwise vocals -30, drums -22, bass -40 dB.
2. The cut is at most 20 dB. Only narrowing, never widening.
3. Guitars and keys keep their width.

**Effect** on the audience recording:

| Measure | Before | After | Studio |
|---|---|---|---|
| Mix side re mid | -6.9 dB | -11.8 dB | -12.8 dB |
| Mono loss | -0.94 dB | -0.32 dB | -0.28 dB |
| Vocal stem side re mid | -3.4 dB | -25 dB | — |

The vocal stem's 2-8 kHz texture also moved toward the studio vocal's.

## 3. De-ess (`deess`), drum transients (`transients`), width (`widen`)

**De-ess.** A dynamic de-esser on the vocal stem, working in the 4.5-11 kHz band.

1. Compare the band envelope (1 ms attack / 60 ms release) with the full envelope (5 / 120 ms).
2. The threshold is the 80th percentile of their dB difference on active passages, so only the brightest 20 % of
   moments are touched.
3. Ratio 3:1, maximum cut 6 or 9 dB.
4. Only the sibilance band is reduced; the rest of the voice is untouched.

Broadcast vocals are often presence-boosted: the vocal carried 61-68 % of the 3-10 kHz energy.

**Drum transients.** The dB difference between a fast (0.3 / 40 ms) and a slow (15 / 150 ms) envelope of the drum
stem is scaled (`amount`) and capped (`max_db`). It is light when de-limiting already lifted the hits.

**Width.** The guitar/keys stem gets +3 dB of side signal above 150 Hz, after polarity repair. Never on a mono or
near-mono source: that would be synthetic stereo.

It is also off when the drum stem's median L-R correlation is under 0.8. That is a diffuse, room-miked recording,
for example an audience pair, and it is already wide. More side there only adds room and costs mono compatibility.
One such recording measured:

- drums 0.65 and vocal 0.32;
- side only 3.8 dB under mid;
- mono loss -1.6 dB.

Broadcast and board mixes measure 0.98-1.0 on the drums.

## 4. Per-stem tone match (`stem_match`)

**Why.** Global EQ cannot cut a harsh vocal without dulling cymbals, or lift a buried kick without thickening the
guitars. Each instrument is compared with the same instrument in the reference.

**How.**

1. For each stem and each section, take the long-term spectrum on a 1/6-octave grid, normalised at the stem's core
   band:

   | Stem | Core band |
   |---|---|
   | vocals | 300-1000 Hz |
   | other | 300-1000 Hz |
   | drums | 100-5000 Hz |
   | bass | 60-250 Hz |

2. Compare it with the reference stem shape, smooth over half an octave, and take `alpha` = 60 %.
3. Boost only where the stem carries a real share of that band: no boost below a 5 % share, full boost above 30 %.
   Lifting another instrument's bleed would be wrong.
4. Caps: +3 dB (+4 dB below 150 Hz) and -6 dB. The vocal's 1-8 kHz presence may rise +6 dB, because a studio vocal is far brighter there than a live one, but that boost is still scaled by how much of the band the vocal owns. Bass and kick are never cut below 120 Hz.
5. Render each section's curve as an 8191-tap linear-phase FIR; sections crossfade over 1.5 s.

**Why core-band normalisation.** An earlier version normalised to the stem's total energy. A very bright live vocal
then made the algorithm *raise the mids* (+4 dB at 1 kHz), which sounded nasal. Normalising at the core band turns the
same deviation into a cut of the harsh top.

## 5. Stem balance (`balance`)

Each stem's share of the mix moves toward the reference share, per section, with crossfaded gains:

| Image | `fraction` | `limit_db` |
|---|---|---|
| Mixed (board or broadcast) | 50 % | ±3 dB |
| Room-miked | 75 % | ±6 dB |

A room-miked recording is far from the bass amp and close to the crowd. On one, the bass stem sat 20 dB under its studio share and "other" (guitars plus room and crowd) 6 dB over. Letting the balance move further gives a defined bass instead of room boom.

Example: a broadcast had the bass 6-8 dB under its studio share and got +2.8 / +3 dB. In its second half, the
guitar layer was 3.5 dB over the vocal; the guitars got -2.9 dB and the vocal +2 dB.

**When the performance has a different arrangement.**

- A stem below -30 dB re mix is absent and left alone. Raising it only lifts bleed.
- With no drums at all, the performance is acoustic or stripped down, and the step is off. The reference's ratios
  come from the full band.
- Example: a song played with voice and one instrument against its full-band studio version. Drums sat at -53 dB
  and bass at -65 dB re mix, and the drum stem held only the applause after the song. The balance step lowered the
  lead vocal by 4.4 dB and raised the applause by 6 dB.

## 6. Dynamics per stem (`dynamics`)

**Why per stem.** After de-limiting, the peaks come back mostly on drum hits: 97 % of the top 0.1 % peaks in the
example. A master limiter would then duck the whole mix on every hit, which is audible pumping.

1. **Drum peaks.** The drum stem alone goes through a 1 ms-lookahead limiter. It shaves the first milliseconds of
   each hit; the body of the hit, which carries the punch, stays. The ceiling is set relative to the drum stem's own
   loudness:
   - +11 dB on a limited source, where the de-limiter brought the peaks back;
   - +15 dB on an open one, whose drum peaks are natural. At +11 dB, an open audience recording lost more than 1 dB
     on 18 % of the time (up to 11 dB), and 1.8 dB of 3 s crest.
2. **Compressors per stem.**
   - Ratios on a limited source: bass 3:1, vocal 3:1, guitars 2.5:1. On an open one: 2:1, 2:1, 1.5:1.
   - Feed-forward and stereo-linked, with a 6 dB soft knee.
   - The threshold is a percentile of the detector level (70, 60 and 60), so it adapts to the file.
   - Make-up gain restores the RMS, so only the shape of the level changes.
3. **Bus glue.**
   - Limited sources: 1.8:1, 25 / 250 ms, about 0.3 dB average. It only binds the mix together.
   - Open sources: none.

## 7. Overall tone match (`match`)

With a same-song reference whose passage `analyze` found (`match.measure_section`), steps 4, 5 and 7 measure on
that passage only and apply one fixed curve to the whole song. See `diagnosis.md`, "Like with like".

After the stems are summed, the long-term spectrum (1/6 octave, normalised at 0.5-2 kHz) moves `alpha` = 85 % toward
the reference profile:

- the correction is smoothed over half an octave and kept between -8 and +6 dB;
- the correction fades out below 45 Hz;
- no boosts above the cut-off;
- each section gets its own curve, rendered as a linear-phase FIR.

The per-stem match fixes local problems; this step sets the overall balance. Without it, removing bleed from vocal
and guitar stems had left the whole mix thinner (80 Hz still 2-3 dB under) and darker above 12 kHz.

## 8. Band extension (`band_ext`)

**Why.** Lossy codecs remove everything above about 15-17 kHz. The music sounds closed, especially on good
headphones.

**How (SBR-like, `dsp.hf_extend`).**

1. Take the band just below the cut-off, [fc - Δ, fc] with Δ = min(fc/4, 19.5 kHz - fc).
2. Shift it up by Δ with a single-sideband (Hilbert) shift. A shift, not a pitch change, so cymbals and sibilants
   stay aligned with their hits.
3. Repeat until about 19.5 kHz.
4. High-pass at the cut-off and roll off from 17.5 kHz.
5. Replace half of its power (`noise_mix` 0.5) with a noise-like copy that has the same envelope: STFT magnitude
   smoothed over about 400 Hz, random phase shared by both channels, so panning and timing stay.
6. Scale it so that [fc + 300, fc + 2500] matches the reference profile's power in that band (`below_ref_db`
   offsets it). Without profile data, it continues the slope of the octave below at -6 dB.

**Why the noise-like half.** A shifted band keeps the sparse, peaky texture of what it was shifted from. Real air,
from cymbals and breath, is dense. On the audience recording, the 13-19 kHz flatness was:

| Rebuilt top | Flatness |
|---|---|
| Shifted copy of the raw codec band | 0.10 |
| Shifted copy after codec repair | 0.24 |
| After codec repair, with the noise-like half | 0.31 |
| Studio master, for comparison | 0.39 |

The level matched the studio master within 1 dB in all three.

**Limits.** This is synthetic "air" built from the band below. It is not the original recording's top. Keep it at or
below the reference level.

## 9. Bass to mono

Side below 100 Hz is removed (4th-order high-pass, zero phase). The low end tightens, mono playback improves, and no
low-frequency phase fights reach the limiter.

## 10. Loudness contour (`macro`, open recordings)

**Why.** The song's quiet intro, its breaks and its build-ups are part of the performance. On an open recording
(LRA 10 LU), every step above nudged them:

| Change | Loudness range left |
|---|---|
| The limited-source dynamics | 6.4 LU |
| Light compression | 7.5 LU |
| No compression at all | 8.0 LU |

Even without compression, 2 LU went. The tone EQ and the stem balance change how loud a bright guitar intro is next
to a bass-heavy chorus.

**How.**

1. Take the K-weighted 3 s level of the input and of the processed music every 0.5 s, each relative to its own
   median over the music.
2. Smooth the difference over 4 s and clip it to ±6 dB (`max_db`).
3. Apply it as a slow gain.

At this time scale nothing changes in the hits or the tone, only how loud each part of the song is next to the
others. The overall level is left to the master.

**Off for limited sources:** their contour is the limiter's, not the band's.

## Repairs on the raw mix (both paths)

| Repair | When |
|---|---|
| 20 Hz high-pass (2nd order) | Always |
| Mains notches (Q 35) at the flagged harmonics | Hum detected |
| ffmpeg `adeclip` | Clipping detected |

Stems receive the same high-pass and notches.

## Loudness (`master`)

Music gets the **loudest** integrated loudness, searched from -8 LUFS downward in 0.5 dB steps, at which the
lookahead limiter would reduce more than 1 dB for at most a set share of the time, and more than 2 dB for at most
another:

| Source | Above 1 dB | Above 2 dB | Result |
|---|---|---|---|
| Limited | 8 % | 1.5 % | As loud as a modern master without audible limiting |
| Open | 2.5 % | 0.5 % | The limiter stays almost idle; the dynamics survive |

The limiter is 4x inter-sample-aware, with a 1.5 ms ramp, 120 ms release and a ceiling of -1 dBTP.

On the open example, the open-source criteria kept 14.5 dB of crest (source 14.8) at -13 LUFS (source -15.4).

**Crest guard (open sources).** The plan's `min_crest_db` is the original's 3 s crest minus 0.5 dB. At each
candidate level that meets the limiter criteria, the limited music's crest is computed, and the level steps down
0.5 dB until it holds. On the test song, -13.0 LUFS gave 14.26 dB against a floor of 14.30, so the master took
-13.5 LUFS (14.41 dB). This is what verify's dynamics check measures, so it no longer needs a hand edit.

**How far down.**

- Limited sources stop at -14 LUFS.
- Open sources may go down to the original's own loudness minus 0.5 dB (verify's "not quieter" limit), at least
  -14 and at most -20 LUFS. One audience recording sat at -25 LUFS with peaks 22 dB over its loudness. At -14 LUFS
  the limiter still worked above 2 dB for 0.6 % of the time and took 1.1 dB of crest.
- The crest guard asks only for what the music brings to the master (its crest before the limiter, -0.2 dB).
  Otherwise compression earlier in the chain would send the search to its floor for nothing.

- Why this measure: "busy time" counts even 0.01 dB of release tail, and "maximum reduction" is set by one freak
  peak. Time spent above 1 and 2 dB is what you can hear.
- One correction pass compensates the loudness the limiter itself removes.
- The example broadcast landed at -10.5 LUFS (source -9.6), with 7 % of the time above 1 dB of reduction.
