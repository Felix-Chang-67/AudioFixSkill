# Pitfalls already met once

Each of these happened while building and validating this pipeline. The defaults now avoid them; keep them in mind
before changing a default.

## Process

1. **Being too conservative reads as "nothing changed".**
   - What happened: a first version moved the tone only halfway toward the reference, lifted the drums by about
     2 dB and delivered 2.4 dB quieter than the source. The listener's verdict: "a bit better, but not a difference
     you hear at once".
   - What the pipeline does now: the music path tracks the reference closely (per stem 60 %, overall 85 %) and keeps
     loudness near the source's.
   - Offer the natural settings when someone wants subtle.
2. **Never trust a loud-vs-quiet comparison.** Louder sounds better. `AB.wav` matches loudness per pair.
3. **"Sounds better" is not yours to claim.** Give numbers and let the user listen.

## Dynamics

4. **An off-the-shelf "limiter" can be a compressor.**
   - What happened: pedalboard's `Limiter`, a wrapper of JUCE's, hides a -10 dB, 4:1 compressor before a hard
     clipper. On a loud source it squashed everything again and landed at -7.5 LUFS.
   - The pipeline uses its own lookahead true-peak limiter.
5. **A lookahead limiter must never step its gain.**
   - What happened: taking `min(box average, lookahead minimum)` made the gain jump 1.5 ms before peaks, which
     clicks.
   - The fix is a box average of a forward minimum. It provably never exceeds the needed gain: every value averaged
     at sample i is a minimum over a window that contains i.
6. **"Busy time" and "maximum reduction" are the wrong limiter criteria.** Busy time counts 0.01 dB release tails;
   maximum reduction is set by one freak peak. Use the share of time above 1 dB and above 2 dB.
7. **De-limiting brings the drum peaks back, and a master limiter then pumps the mix.**
   - Measured: 97 % of the top peaks came from the drum stem.
   - Control peaks on the drum stem alone; compress bass, vocal and guitars separately.

## Tone

8. **Per-stem EQ normalised to total energy raises the mids of a bright stem.** A harsh live vocal got +4 dB at
   1 kHz, guitars +6 dB at 2 kHz. Normalise each stem at its own core band and cap boosts.
9. **Per-stem cleanup thins the whole mix.** Removing bleed from the vocal and guitar stems took low end with it:
   80 Hz ended up lower than before. Add a global match after the stems are summed; never cut bass and kick below
   120 Hz.
10. **The profile defines "good".** A wrong-genre reference pulls the music toward that genre. Build one from the
    artist when possible.

## Stereo

11. **Anti-phase can be partial.**
    - What happened: a guitar layer was reversed only between 150 Hz and 2.5 kHz and in phase above 6 kHz, and only
      while that part played.
    - Flipping a whole channel or stem would have broken the highs and the in-phase passages.
    - Rotate only coherent anti-phase bins.
12. **A mix-level anti-phase number cannot tell you which instrument it is.** The vocal of the same broadcast showed
    3.4 % "anti-phase" from bleed. Decide per stem, with a threshold.

## Segmentation

13. **Section cuts must follow the recording chain, not the arrangement.** Tonal-shape changes inside music are the
    music. Cut only where the stereo image of the chain jumps.
14. **Rhythm-section energy cannot separate singing from talk.** Breakdowns and ballads have little drums or bass.
    Use pauses (sung: about 0.05, spoken: about 0.45) and Whisper word density.

## Speech

15. **The vocal stem also catches the crowd's shouting.**
    - What happened: a voice-activity detector on the vocal stem called 97 % of an MC "speech", so crowd taming
      stayed off while the gaps were in fact 5-12 dB *louder* than the talk.
    - Use Whisper word timings: crowd shouting has few or no words.
16. **Separation can lose the voice entirely.** A robotic or very reverberant voice landed 95 % in "other". Detect it
    (`vocal_share` < 0.3) and work on the mix.
17. **Whisper's language ID can be confidently wrong on degraded voices** (English taken for Welsh at 0.77). Decide
    the language by decoding a sample in the top candidates.
18. **Counting all words punishes crowd taming.** Fewer crowd words is the goal. Count only words inside the talker
    mask.
19. **Neural denoisers destroy live speech.** DeepFilterNet3 and Facebook's Denoiser removed most of the words of a
    PA'd, reverberant MC (Whisper confidence collapsed). Reference-free quality scores such as SQUIM still praised
    them. Gentle stationary denoise only.
20. **A PA'd voice is supposed to be thinner than studio speech.** Matching the speech spectrum fully boosted
    125-500 Hz by +6 dB and brought up the room. Cap speech boosts at +3 dB.

21. **A Whisper word count is a noisy measure of intelligibility.**
    - What happened: on a Japanese MC, the restored file "lost" 7 % of its words (165 → 153) while confidence
      rose. The transcripts showed the restored version had recovered a whole phrase and corrected two misheard
      words.
    - Two runs on the same file with the same settings already differ by several %: Whisper's temperature fallback
      samples at random. Japanese word tokens also merge and split.
    - Turning the fallback off is worse: Whisper then loops. One restored MC came back with the crowd's "oh"
      repeated 110 times in place of 30 s of talk. Counted naively, that looked like a 43 % gain in recognised words.
    - Verify now keeps the fallback but seeds it (reproducible), drops any loop that survives it with Whisper's own
      compression-ratio test (> 2.4), and compares the sum of word probabilities with a 10 % margin. Real speech
      damage, as from the neural denoisers above, collapses that sum. Read the transcript before quoting a large
      gain.
22. **"Boost only where the voice dominates" needs to know where the voice is.**
    - With stems, the vocal stem shows where the voice is.
    - On the mix alone, every band looks like "voice". A synthetic MC whose voice stops at 5.5 kHz got a +3 dB "air"
      boost on the crowd above it.
    - The voice's share is now estimated as the talk minus the quieter half of the gaps.
    - For the same reason, the voice-tone check is skipped when the voice is the mix. Denoising the background moved
      the measured "voice" away from its target although the voice EQ was right.
23. **Without stems, an MC is taken for music.** The content map needs the drum and bass stems. Pass
    `--content speech`; `analyze` now warns.

## Numerics

24. **Floating-point smoothing can go slightly negative.** The log of a negative is NaN, and NaN gains silently ruin
    audio. Clamp before `log10`.

## Learned from an open recording and a second assistant

The same audience recording was restored twice with this skill: once by Claude, once by ChatGPT.

25. **Settings tuned on a limited broadcast flatten an open recording.**
    - What happened: the defaults came from a limited broadcast. On an open recording (LRA 10 LU, PLR 15.5 dB), the
      drum limiter at +11 dB shaved more than 1 dB for 18 % of the time. The firm compressors and a loud master
      took the range to 6.4 LU and raised the quiet intro by 4-5 dB against the choruses.
    - The only dynamics check then was the 3 s crest, and it passed.
    - What the pipeline does now:
      - open recordings get light dynamics and the contour step;
      - their limiter stays almost idle;
      - verify checks the loudness range (LRA) of open recordings.
26. **Changing the tone changes the dynamics too.** With no compression at all, the tone EQ and stem balance alone
    still took 2 LU off. A bright intro gains loudness from a presence boost that a bass-heavy chorus does not get.
    Hence the contour step after all processing, not just lighter compressors.
27. **A "safe" result can be quieter than the original.** The other assistant's version came out at -16.0 LUFS
    against a -15.4 LUFS source, with the rebuilt top 7-10 dB under the reference. For open recordings, verify now
    warns when the result is quieter than the original (pitfall 1 again).
28. **Hand-tuning makes results depend on the assistant.**
    - The two versions used the same method: their stem balances agreed within 0.7 dB. They still differed by 3 dB in
      loudness, 2.6 LU in loudness range, and up to 7 dB in the rebuilt top, because of different hand edits.
    - The decisions that were hand-tuned are now rules: limited or open, partial anti-phase, room-miked width.
    - Edit only for a WARN, a measurement the rules missed, or the listener's words, and report every edit.
29. **Anti-phase in a few bands slips past a window-correlation test.**
    - What happened: a guitar stem reversed only below 2.5 kHz showed 0 % of its energy in anti-phase 4 s windows.
    - The repair's own detector measures coherent anti-phase energy per bin, and now decides together with the
      stem's mono loss.
30. **Widening an already-wide recording.**
    - A room-miked audience pair is wide by nature: its drums measured an L-R correlation of 0.65, its side only
      3.8 dB under mid. Adding width there costs mono compatibility and adds room.
    - Width is now added only to mixed images, with drums at a correlation of 0.8 or more.

## Learned from comparing with the studio master of the same song

31. **Right level, wrong texture.**
    - What happened: the rebuilt top matched the studio master's level within 1 dB, and still sounded fake and
      harsh to the listener.
    - Measured on the same 30 s against the studio master, as spectral flatness:

      | Band | Studio master | Restored |
      |---|---|---|
      | 13-19 kHz | 0.39 | 0.10 |
      | 8-12.5 kHz | 0.42 | 0.15 |

    - The cause: a low-bitrate codec leaves the band under its cut-off sparse and peaky. The tone match boosted it
      and the band extension copied it up.
    - Now: codec repair (Apollo) refills the band first, and half of the rebuilt top is a noise-like copy. That
      brought the flatness to 0.29 and 0.31.
32. **The artist's average is not this song.** The 27-song profile left 100-400 Hz 5-12 dB under the song's own
    studio master, which has an unusually loud bass. The song's 30 s preview is now the recommended reference
    (`--itunes-song`).
33. **A room-miked vocal is as wide as the room.**
    - What happened: the vocal stem of an audience recording had its side only 3 dB under mid. In the studio master,
      it was 31 dB under.
    - Tone matching cannot fix width. Per-stem width matching on room-miked images can: vocal, drums and bass
      narrowed, guitars kept.
34. **Separation limits how far a stem can be pushed.**
    - The live vocal stem lacked 7-13 dB of the studio vocal's 2.5-6 kHz presence.
    - Boosting that much would lift the cymbal bleed in the vocal stem too. The vocal's presence boost is capped at
      +6 dB and scaled by how much of each band the vocal owns. The rest of the gap is information the recording
      does not hold.
35. **A preview is one passage; a live song is many.**
    - Matching the whole live song's average spectrum to a 30 s chorus preview over-boosted the lows by 5-7 dB in
      the chorus itself. The live recording's low end varied by 8 dB between passages.
    - Now the preview's passage is found in the recording by chroma, and the tone is measured there.
36. **"The same song" can be a live version.** The song search returned the album track and a live-album recording
    of it, whose vocal width differed by 20 dB. The filter now keeps the studio version.
37. **A codec low-pass is not always a brick wall.**
    - What happened: an MP3-style encoder rolled off over 1.5 kHz (12.7 kHz: -35 dB, 14.3 kHz: -53 dB re mid), and
      the wall detector saw a 7.7 dB "edge". Codec repair and band extension stayed off, although the band under
      the cut-off was the sparsest seen (flatness 0.067).
    - Now: codec damage is judged by that texture, and the band extension starts below the transition band.
38. **A quiet sung passage is not an MC.**
    - What happened: a song had three passages without drums or bass, 76 s of its 249 s. All three were labelled
      speech and given the MC treatment.
    - Singing pauses between phrases, here 36-42 % of frames, above the 25 % rule. Whisper also wrote the sung
      lyrics down as words, about 1 word/s, so asking it did not help. Whisper found no talk in those passages
      afterwards (0 talker words), which flagged the error.
    - Now: when the user says what the file is, `--content music` or `--content speech` decides, and the automatic
      split is kept for files that hold both.
39. **Only the assistant that wrote the skill used the song's own studio version.**
    - What happened: another assistant ran the same skill on the same kind of recording and never built a same-song
      reference.
    - Why: `AGENTS.md` and the quick start went straight to `auto`, which silently used the generic profile. The
      advice sat in one table of `SKILL.md`. A sandbox without internet could not have fetched the preview anyway.
    - Now: `analyze` / `auto` look the studio version up themselves from the tags, or `--artist` / `--title`. When
      they cannot, a `[next]` line asks for the studio track.
40. **A cosine of chroma vectors matches anything.**
    - What happened: a looked-up preview "matched" a different song at 0.91, in a passage that was mostly the
      closing silence. Unit-norm chroma is never negative, so any two frames have a high cosine: 0.82-0.91 for every
      pair tested.
    - Now: each frame has its mean removed before correlating, and a match must stand out in its key. The same
      song scored 0.46-0.67, other songs at most 0.24; the threshold is 0.35.
41. **A stripped-down performance is not the studio arrangement.**
    - What happened: a song was played with voice and one instrument, while its studio version is a full band. The
      balance step pulled the stems toward the studio ratios. It lowered the lead vocal by 4.4 dB and raised the
      "drum" stem by 6 dB, which held only the applause after the song.
    - Now: a stem below -30 dB re mix is left alone, and without drums the balance step is off.
42. **The loudness search had a floor the crest guard could not live with.**
    - What happened: a -25 LUFS audience recording with peaks 22 dB over its loudness could not keep its crest
      above -14 LUFS, where the search stopped. Every assistant would have hit the WARN, and the documented action
      (stricter limiter criteria) could not clear it.
    - Now: open sources search down to the original's loudness minus 0.5 dB, never below -20 LUFS.
43. **A rebuilt top louder than the studio version is heard on the vocal.**
    - What happened: two renders identical below 12.5 kHz differed only in the rebuilt top, which was 2-3 dB louder in
      one of them. That one sounded "a little digital, the vocal scattered"; the other did not.
    - The single-gain extension let the copies above the calibrated band follow the band they came from: 5-10 dB over
      the studio roll-off at 16-19 kHz. On another song, it started above a moving cut-off and left a notch.
    - Now: the top is filled per 20 s and 250 Hz only up to the reference's own level, from under the lowest cut-off.
44. **PLR alone called a dense live song "limited".**
    - What happened: a PLR of 10.9 dB, just under 11, switched on de-limiting, firm compression and a loud master on a
      song with a 13.6 dB crest and a 6.7 LU range, where the De-limiter found 0.1 dB of limiting. Its range went to
      3.1 LU.
    - Now: a low PLR counts only with a 3 s crest under 12.5 dB.
