# Quality audit - where the tool picks badly

## Headline findings
- 3 promos were rejected by the judge by only 1 point - the thresholds ({'hook': 7, 'cliffhanger': 7, 'standalone': 6}) decide a lot; review them in rejected.json.
- 74 good-sounding ideas died on mechanical rules (duration/continuity), not on quality - candidates for auto-repair (trim/extend to snap into 20-60s).
- 6/22 promos trigger 'dead_air'.
- 4/22 promos trigger 'crowded_cast'.
- 12 episodes have a high-intensity 30s window (punctuation + tone tags) that no promo covers - possible misses (proxy metric, check by eye).
- 157 discovered moments rated intensity >= 8 did not reach the final list (71 never turned into a promo, 86 dropped by validation/judge/de-dup).
- Critic flagged 6/12 sampled promos (even among the top-ranked); most common failure: [('SLOW_SETUP', 4), ('RESOLVED_ENDING', 3)].

## Pipeline funnel
- Kept 22, rejected 86: validation: not continuous (gap): 31, validation: duration out of range: 23, validation: not continuous (episode jump): 20, judge: below threshold: 11, validation: overlap/order: 1

## Distribution
- Duration buckets: {'30-40s': 7, '20-30s': 7, '50-60s': 2, '60-70s': 1, '40-50s': 5}
- Episodes with no promo: [2, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 20, 21, 22, 24, 25, 26, 28, 29, 34, 35, 36, 37, 39, 40, 41, 42, 43, 46, 47, 49, 50]
- Start position within episode: {'first': 1, 'middle': 7, 'last': 14}

## Per-promo mechanical flags
- X11-1: dead_air: 5.2s with nothing happening
- E12-p1: crowded_cast: 4 speakers (hard to follow cold)
- E15-p1: reply_fragment_open: starts on 'No.' (needs the line before it)
- X15-1: crowded_cast: 5 speakers (hard to follow cold)
- E19-p1: reply_fragment_open: starts on 'Your Grace.' (needs the line before it)
- X23-1: dead_air: 6.8s with nothing happening
- X30-1: dead_air: 5.0s with nothing happening
- X32-1: crowded_cast: 4 speakers (hard to follow cold)
- X33-1: dead_air: 5.7s with nothing happening, crowded_cast: 4 speakers (hard to follow cold)
- E44-p1: dead_air: 4.6s with nothing happening
- E45-p1: dead_air: 4.5s with nothing happening

## Adversarial critic (sample)
- Verdicts: {'good': 6, 'fixable': 6}; tags: {'SLOW_SETUP': 4, 'RESOLVED_ENDING': 3, 'WEAK_HOOK': 1}
- E12-p1 [fixable] ['SLOW_SETUP']: Takes too long to get to the core blackmail threat, dragging down the pacing in the middle. -> fix: Start at Clarissa's line 'Fail, or refuse, and your husband hangs from the gallows' and trim the dialogue leading up to it.
- E45-p1 [fixable] ['RESOLVED_ENDING']: The ending dialogue repeats the 'not lawful wife' beat twice, softening the impact of the cliffhanger. -> fix: Cut the final line at [01:48.80] and end right as the High Priest delivers the condemnation, before the close-up repeats the beat.
- X23-1 [fixable] ['SLOW_SETUP']: The first 20 seconds of lingering door-knob suspense is too slow for a vertical short-drama promo hook. -> fix: Cut directly to the noblewoman entering at [00:03.50], starting the promo with the confrontation instead of the hallway build-up.
- X33-1 [fixable] ['WEAK_HOOK']: Starts with a weeping maid rather than an immediate narrative collision or confrontation. -> fix: Start at E34 00:01.20 with the Judge reading the accusation against Lady Juliana.
- X13-1 [fixable] ['RESOLVED_ENDING', 'SLOW_SETUP']: Ends on a tender romantic reunion rather than a high-stakes cliffhanger, killing the ReelShort hook. -> fix: Cut right after the handcuffs unlock at 00:15.50 before the romantic hand-holding begins.
- X03-1 [fixable] ['RESOLVED_ENDING', 'SLOW_SETUP']: The clip lingers on a slow emotional acceptance and fade before cutting to the insults, neutralizing the punch of the cliffhanger. -> fix: Cut the slow fade in the middle and end directly on Juliana's face as Clarissa's insult lands.

## Judge stability
- {'clips_rescored': 12, 'mean_abs_score_change': 0.81, 'share_changed_by_2_or_more': 0.17}

## Possible misses (proxy: most intense 30s per episode, not covered by any promo)
- Ep 28 00:20.00-00:50.00 (intensity 30.0)
- Ep 11 00:55.00-01:25.00 (intensity 26.0)
- Ep 3 00:50.00-01:20.00 (intensity 24.5)
- Ep 16 00:20.00-00:50.00 (intensity 24.5)
- Ep 42 00:05.00-00:35.00 (intensity 23.5)
- Ep 36 01:30.00-02:00.00 (intensity 23.0)
- Ep 43 00:00.00-00:30.00 (intensity 23.0)
- Ep 29 00:05.00-00:35.00 (intensity 21.0)
- Ep 47 00:55.00-01:25.00 (intensity 21.0)
- Ep 21 00:00.00-00:30.00 (intensity 20.0)

## Strong moments discovered but not in the final list
- E01-m4 Ep 1 01:57.00-02:08.50 [OTHER, 10, candidate_dropped]: Gerald publicly scorns Juliana to her face, declaring she has nothing to offer and calling Clarissa the true jewel.
- E02-m5 Ep 2 01:33.40-01:50.00 [REVELATION, 10, candidate_dropped]: Juliana delivers a scathing public indictment of Gerald's honor, declaring that he has shown everyone exactly what his word is worth.
- E11-m6 Ep 11 01:33.20-01:46.80 [REVELATION, 10, candidate_dropped]: The constable pulls Gerald's gold pocket watch directly out of the lining of Alexander's coat, leaving Juliana in sheer horror and Gerald victorious.
- E13-m6 Ep 13 01:43.00-01:58.88 [REVELATION, 10, candidate_dropped]: Amidst severe lacerations and blood, Juliana's fingers brush against and successfully unearth the lost emerald signet ring.
- E18-m6 Ep 18 02:51.80-03:12.00 [REVELATION, 10, candidate_dropped]: With a face hardened into lethal fury, Alexander vows a chilling vengeance, promising they will not take anything else.
- E22-m4 Ep 22 02:05.80-02:11.00 [SUSPENSE, 10, candidate_dropped]: The Dowager Duchess convulses and gasps violently for air while Juliana desperately demands to know who has been treating her, ending on a tense cliffhanger.
- E23-m5 Ep 23 01:52.20-02:11.50 [SUSPENSE, 10, not_used]: Heavy footsteps echo outside the room just as Juliana is mixing the antidote, and the door handle slowly begins to turn.
- E28-m5 Ep 28 01:33.80-01:41.20 [REVELATION, 10, not_used]: Catherine reads the handwritten name 'Juliana Ashcroft' inside the herbarium, exposing Juliana's true noble identity.
- E29-m7 Ep 29 01:49.00-01:52.80 [TURNING_POINT, 10, candidate_dropped]: Sobbing turns to steely resolve as Juliana wipes her tears and swears that they will never take Alexander from her.
- E34-m6 Ep 34 02:46.80-03:10.00 [SUSPENSE, 10, not_used]: Standing unbroken before the court, Lady Juliana comforts the crying crowd, while the judge rises to his full height to pronounce the final sentence.
- E35-m6 Ep 35 02:17.00-02:37.50 [THREAT, 10, candidate_dropped]: Alexander rages that his ambush was no accident, dropping the chilling revelation that the men behind his attempted murder currently rule Ashwood.
- E36-m5 Ep 36 03:08.00-03:20.75 [SUSPENSE, 10, candidate_dropped]: Alexander makes a chilling final vow to the night air, declaring that their dead Duke is coming home before the wedding begins.
- E41-m6 Ep 41 01:20.50-01:25.50 [THREAT, 10, candidate_dropped]: An ear-splitting explosion blasts open the cathedral doors, sending the congregation into panic.
- E42-m6 Ep 42 01:13.80-01:17.80 [THREAT, 10, candidate_dropped]: Alexander vows to Rupert that they will now settle everything that was stolen, resting his hand menacingly on his sabre.
- E50-m1 Ep 50 01:10.50-01:20.00 [REVELATION, 10, candidate_dropped]: Agnes discovers Catherine and Father Thomas locked in a passionate embrace in the abandoned wing and stumbles away in terror with their lethal secret.

## Next step: ground truth
- Fill `human_verdict` (good/ok/bad) in audit_labels.csv for the listed promos, then run `--labels out\audit_labels.csv`. Until then every number above is the model grading itself, and mechanical flags and proxies are not ground truth.

## Known structural limits
- Text only: acting, music, visuals and pacing of the real video are invisible, so 'hook in first 3 seconds' is judged from script lines.
- The same LLM family proposes, judges and critiques, so shared blind spots survive every check.
- Thresholds (7/7/6) and weights are my choices, not tuned against real post performance.