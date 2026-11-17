# Tuning from listener feedback

First run `audiofix.py investigate --work WORK --complaint "<the listener's words>"` (SKILL.md step 8). It measures
the likely causes and renders a variant for each. This table explains the knobs it uses, and the ones to try when the
listener asks for something specific.

Change one thing at a time, re-render, and send a new `AB.wav`. Overrides go on the command line (they are saved in
`plan_used.json`) or into `plan.json`:

```bash
python3 scripts/audiofix.py restore --work WORK --set music.match.hi_db=3 music.delimit.depth=1.5 \
  --why "the listener: boomy and pumping"      # required: restore refuses an edit without its reason
```

## Songs

| The listener says | Likely meaning | Edit |
|---|---|---|
| Too much bass, boomy | The low boost from the match/balance is too big for their speakers | `music.match.hi_db=3`; or `music.balance.limit_db=1.5` |
| Still thin | The reference wants more low end than the match was allowed to give | `music.match.hi_db=8` (careful); build an artist profile |
| Muddy | 200-500 Hz still high | `music.match.alpha=0.95` (closer to the reference) |
| Harsh, sibilant | Bright vocal or cymbals | `music.deess.max_cut_db=12`; `music.match.alpha=0.95` |
| Dull, lost its sparkle | Top cut too much, or no band extension | `music.match.lo_db=-4`; `music.band_ext.enabled=true`; `music.band_ext.below_ref_db=-2` |
| Fake, metallic or harsh highs | The codec's sparse texture, boosted and copied up | Make sure codec repair ran (`analyze --codec-repair on`); `music.band_ext.noise_mix=0.7`; then `music.band_ext.below_ref_db=3` |
| Fizzy top, "digital" highs | The rebuilt top is too loud | `music.band_ext.below_ref_db=3`; or `music.band_ext.enabled=false` |
| A faint digital feel, a whistle, or a digital, scattered vocal | Codec tones (whistles), steady or on and off, lifted and copied up by the restoration; notes the codec copied up into the highs (band replication); or a rebuilt top louder than the reference | Check the `whistles` items in `review.md`, and `[repair] notched ... codec tone(s)` and `[repair] cut the tones band replication copied up` in the log: all three are removed by default. If a whistle is still heard, run `investigate` with the listener's words: it measures what is left in the render and renders a variant that notches it. Then read `result_minus_target_db` in the `[music] top rebuilt` line, and try `music.band_ext.below_ref_db=3` or 4. A 50 kb/s rip had four steady whistles at 8.4-10.5 kHz, raised 10 dB and copied to 19.1 kHz, and three more that came and went at 6.7-7.8 kHz. Another render, 2-3 dB brighter at 12.5-21 kHz, was heard as a digital, scattered vocal |
| Not clean, unfocused, the vocal floats, "I can hear the room" | A room-miked image | `music.width_match.enabled=true` (default on room-miked recordings); narrower: lower the targets, e.g. `music.width_match.targets='{"vocals":-35,"drums":-26,"bass":-45}'` |
| Should sound like the album | The reference is an average, not this song (accepted with `--generic-reference`) | Check `diagnosis.md` → Reference. If it is the generic profile, give the song: `--artist A --title S` (online), or the studio track: `profile --name S --files TRACK --song --out S.json`, then re-analyze with `--music-profile S.json` |
| Squashed, no punch | Master too loud, or not enough de-limiting | `master.over1_max_pct=5 master.over2_max_pct=1`; `music.delimit.depth=2` |
| Pumping, breathing | De-limiting too deep, or drum shaving too hard | `music.delimit.depth=1.5`; `music.dynamics.drum_ceiling_over_lufs=13` (limited sources; open ones already use 15) |
| Too quiet compared with other music | Perceptual target too strict for their use | Limited source: `master.over1_max_pct=12 master.over2_max_pct=2.5`. Open source: `master.over1_max_pct=5 master.over2_max_pct=1` (costs some dynamics) |
| Flat, the intro is as loud as the chorus (open recording) | The song's contour was flattened | `music.macro.enabled=true` (default for open recordings); lighter `music.dynamics` ratios |
| Too uneven, I want it denser, like a studio master (open recording) | The contour step keeps the live dynamics | `music.macro.enabled=false`, then `master.over1_max_pct=5 master.over2_max_pct=1` |
| Too wide, unnatural | Width step | `music.widen.side_db=1.5`; or `music.widen.enabled=false` |
| Still phasey | An anti-phase stem not flagged | `music.antiphase.stems='["other","vocals"]'` |
| Vocal too loud or too quiet | Stem balance | `music.balance.fraction=0.3`, or edit the stem gains by profile |
| Vocal dull next to the album | The studio vocal is brighter at 1-8 kHz | `music.stem_match.vocal_presence_db=8` (the boost still follows how much of the band the vocal owns) |
| Too processed, I want it natural | Strong-remaster defaults | `music.match.alpha=0.5 music.stem_match.enabled=false music.balance.enabled=false music.widen.enabled=false music.delimit.depth=1.5` |
| Watery or swirly artefacts on the voice or cymbals | Separation bleed exposed by big stem changes | `music.stem_match.alpha=0.4 music.balance.limit_db=1.5`; or `music.stem_match.enabled=false` |
| Sounds like a different band | Wrong reference | Build a profile from this artist |

## Talk (MC)

| The listener says | Likely meaning | Edit |
|---|---|---|
| Crowd still too loud between sentences | Cap too loose, or the talker mask includes crowd (fallback detector) | `speech.crowd.cap_db=5 speech.crowd.floor_db=24`; install faster-whisper and re-analyze |
| It sounds dead, no audience | Too much taming | `speech.crowd.cap_db=-1 speech.crowd.gap_db=2`; or `speech.source.ambience_db=-6` |
| Voice too processed, too bright, "radio" | The v2 vocal treatment is too strong for this listener | `speech.mode=gentle` (the earlier, lighter path); or `speech.match.alpha=0.6` |
| Talker should sound like the restored singer | Talk-only file from a show whose songs were restored separately | `speech.match.target=song_vocal speech.match.chain_file=SONGWORK/vocal_eq.json` |
| Broadcast MC sounds squashed | Talk-only broadcast: de-limiting is not switched on automatically | `speech.delimit.enabled=true` |
| Voice boomy | Low boost or rumble | `speech.highpass_hz.value=110`; `speech.match.hi_db=1` |
| Voice thin, tinny | The match was capped | `speech.match.hi_db=5` |
| Voice muffled | Presence missing | `speech.match.alpha=0.95`; in gentle mode `speech.mode=v2` |
| Synthetic-sounding "s" or fizz on the voice (broadcast rips) | The rebuilt top of the talk | `speech.band_ext.below_ref_db=6`; or `speech.band_ext.enabled=false` |
| Lisping or harsh "s" | De-ess too weak or too strong | `speech.deess.max_cut_db=9`, or `3` for lisping |
| Swirly or underwater voice | Denoise too strong | `speech.denoise.nr_db=4`; or `speech.denoise.enabled=false` |
| Loud and quiet sentences still uneven | Leveler range | `speech.leveler.range_db=9` |
| Talk too loud or too quiet against the songs | Offset | `master.speech_offset_db=2` (louder) or `6` (quieter) |
| Words cut off at sentence ends | Mask too tight | Use the ASR detector; or `speech.crowd.enabled=false` |

## Content map

| Problem | Edit |
|---|---|
| A song treated as MC, or the reverse | Re-analyze with `--content music` or `--content speech`, or edit `segments` in `plan.json` (`[start, end, "music"\|"speech"]`) |
| A feed change not split | Add a cut to `sections` (`[[0, 211], [211, 516]]`) |
| A spurious split | Merge the two entries in `sections` |
