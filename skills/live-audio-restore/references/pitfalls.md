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

## Numerics

22. **Floating-point smoothing can go slightly negative.** The log of a negative is NaN, and NaN gains silently ruin
    audio. Clamp before `log10`.
