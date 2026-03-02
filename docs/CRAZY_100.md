# 100 Crazy Things To Try With `pvx`

These are intentionally bold, sometimes impractical ideas for sound-design exploration.
Use `pvx stretch-budget` and sensible disk limits before launching extreme renders.

## 1-10: Extreme Time-Scale Madness

1. Stretch a one-shot to absurd length:

```bash
pvx stretch-budget one_shot.wav --disk-budget 50GB --requested-stretch 1000000 --fail-if-exceeds --json && pvx voc one_shot.wav --stretch 1000000 --stretch-mode multistage --auto-segment-seconds 0.25 --checkpoint-dir ckpt --resume --output one_shot_blackhole.wav
```
2. Make a 10-minute ambient tail from a single hit:

```bash
pvx voc hit.wav --target-duration 600 --preset extreme_ambient --output hit_10min.wav
```
3. Stretch then shrink in two stages for texture hysteresis:

```bash
pvx chain input.wav --pipeline "voc --stretch 12 | voc --stretch 0.08" --output hysteresis.wav
```
4. Warp tempo with aggressive map swings:

```bash
pvx warp loop.wav --map maps/warp_chaos.csv --crossfade-ms 10 --output loop_chaos_warp.wav
```
5. Build a staircase of tempo regimes:

```bash
pvx conform input.wav --map maps/tempo_stairs.csv --output input_tempo_stairs.wav
```
6. Force tiny windows for intentionally crunchy time stretch:

```bash
pvx voc source.wav --stretch 8 --n-fft 512 --hop-size 64 --output source_gritstretch.wav
```
7. Force oversized windows for smeared macro motion:

```bash
pvx voc source.wav --stretch 8 --n-fft 8192 --hop-size 256 --output source_glassstretch.wav
```
8. Alternate compress/expand blocks in one chain:

```bash
pvx chain groove.wav --pipeline "voc --stretch 0.5 | voc --stretch 2.0 | voc --stretch 0.5 | voc --stretch 2.0" --output groove_pumpfold.wav
```
9. Create a "time collapse" impulse sculpture:

```bash
pvx voc ambience.wav --stretch 0.03 --output ambience_collapse.wav
```
10. Use stateful stream for ultra-long continuity tests:

```bash
pvx stream long_take.wav --output long_take_stateful_20x.wav --chunk-seconds 0.2 --time-stretch 20
```

## 11-20: Freeze + Morph Experiments

11. Freeze the loudest drum transient into a pad:

```bash
pvx freeze drums.wav --freeze-time 0.42 --duration 90 --output drums_freeze_pad.wav
```
12. Freeze a vocal consonant into a synthetic whisper bed:

```bash
pvx freeze vocal.wav --freeze-time 1.03 --duration 40 --output vocal_consonant_pad.wav
```
13. Morph two unrelated worlds with static alpha:

```bash
pvx morph piano.wav subway.wav --alpha 0.5 --output piano_subway_50.wav
```
14. Morph by trajectory map from A to B to A:

```bash
pvx morph a.wav b.wav --alpha controls/alpha_aba.csv --interp linear --output a_b_a_morph.wav
```
15. Use envelope cross-synthesis mode:

```bash
pvx morph source_a.wav source_b.wav --blend-mode carrier_a_envelope_b --alpha 0.8 --envelope-lifter 36 --output env_cross.wav
```
16. Use mask-driven morph with hard spectral selection:

```bash
pvx morph source_a.wav source_b.wav --blend-mode carrier_a_mask_b --alpha 0.7 --mask-exponent 1.5 --output mask_cross.wav
```
17. Keep A magnitude but steal B phase attitude:

```bash
pvx morph source_a.wav source_b.wav --blend-mode magnitude_b_phase_a --alpha 0.65 --phase-mix 0.1 --output phase_cross.wav
```
18. Morph after a freeze for evolving drone color:

```bash
pvx freeze hit.wav --freeze-time 0.2 --duration 60 --output hit_frozen.wav && pvx morph hit_frozen.wav choir.wav --alpha 0.35 --output frozen_morph.wav
```
19. Morph speech into noise and retune it:

```bash
pvx morph speech.wav noise.wav --alpha 0.6 --output speech_noise_morph.wav && pvx retune speech_noise_morph.wav --root C --scale minor --strength 1.0 --output speech_noise_retune.wav
```
20. Build a morph ladder at fixed alpha steps for curation:

```bash
pvx morph a.wav b.wav --alpha controls/alpha_ladder.csv --interp none --output morph_ladder.wav
```

## 21-30: Pitch, Harmony, and Tuning Extremes

21. Build a choir from one voice in one command:

```bash
pvx harmonize lead.wav --intervals 0,4,7,12 --gains 1,0.8,0.7,0.5 --pans -0.4,-0.1,0.1,0.4 --force-stereo --output lead_choir.wav
```
22. Create detuned micro-harmony clusters:

```bash
pvx harmonize lead.wav --intervals 0,3,7 --intervals-cents 0,7,-11 --gains 1,0.8,0.8 --output lead_microcluster.wav
```
23. Hard retune spoken voice to a scale:

```bash
pvx retune speech.wav --root D --scale major --strength 1.0 --output speech_tuned.wav
```
24. Gentle retune with minimal intervention:

```bash
pvx retune vocal.wav --root A --scale minor --strength 0.25 --output vocal_subtle_retune.wav
```
25. Force ratio-based pitch with no stretch change:

```bash
pvx voc input.wav --stretch 1.0 --ratio 7/4 --output input_ratio_7_4.wav
```
26. Extreme semitone dive with formant-preserving mode:

```bash
pvx voc vocal.wav --stretch 1.0 --pitch -19 --pitch-mode formant-preserving --output vocal_deep_formant.wav
```
27. Conform to 19-TET map:

```bash
pvx conform lead.wav --map maps/map_19tet.csv --output lead_19tet.wav
```
28. Conform to just intonation map:

```bash
pvx conform choir.wav --map maps/map_just_intonation.csv --output choir_ji.wav
```
29. Layer two opposite pitch moves:

```bash
pvx chain vocal.wav --pipeline "voc --pitch 7 | voc --pitch -12" --output vocal_pitch_fold.wav
```
30. Pitch-shift then freeze at peak weirdness:

```bash
pvx chain vocal.wav --pipeline "voc --pitch 12 | freeze --freeze-time 0.5 --duration 45" --output vocal_octave_freeze.wav
```

## 31-40: Transients, Stereo, and Layer Split

31. Run hybrid transient preservation on speech at high stretch:

```bash
pvx voc speech.wav --transient-mode hybrid --transient-sensitivity 0.65 --stretch 2.5 --output speech_hybrid_2_5.wav
```
32. Stretch drums with transient-first preset:

```bash
pvx voc drums.wav --preset drums_safe --stretch 1.8 --output drums_safe_1_8.wav
```
33. Mid/side lock for stable wide mixes:

```bash
pvx voc mix.wav --stereo-mode mid_side_lock --coherence-strength 0.95 --stretch 1.4 --output mix_ms_lock.wav
```
34. Intentionally lower coherence for drifting sides:

```bash
pvx voc stereo.wav --stereo-mode mid_side_lock --coherence-strength 0.2 --stretch 2 --output stereo_side_drift.wav
```
35. Split HPSS layers and stretch only harmonics:

```bash
pvx layer full_mix.wav --harmonic-stretch 2.0 --percussive-stretch 1.0 --output full_mix_harm_stretch.wav
```
36. Split HPSS layers and tighten percussion only:

```bash
pvx layer beat.wav --harmonic-stretch 1.0 --percussive-stretch 0.7 --output beat_tight_perc.wav
```
37. Pitch harmonics up while lowering percussion gain:

```bash
pvx layer groove.wav --harmonic-pitch-semitones 5 --percussive-gain 0.7 --output groove_harm_up.wav
```
38. Push transient mode with tiny protect windows:

```bash
pvx voc drumloop.wav --transient-mode hybrid --transient-protect-ms 8 --transient-crossfade-ms 2 --stretch 1.5 --output drumloop_snappy.wav
```
39. Push transient mode with huge protect windows:

```bash
pvx voc drumloop.wav --transient-mode hybrid --transient-protect-ms 80 --transient-crossfade-ms 20 --stretch 1.5 --output drumloop_smearshield.wav
```
40. Compare wrapper vs stateful stream seam behavior:

```bash
pvx stream source.wav --mode wrapper --output source_wrapper.wav --chunk-seconds 0.2 --time-stretch 2.0 && pvx stream source.wav --mode stateful --output source_stateful.wav --chunk-seconds 0.2 --time-stretch 2.0
```

## 41-50: Sidechain Follow and Feature-Driven Control

41. Basic pitch-follow stretch transfer:

```bash
pvx follow guide.wav target.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output target_follow.wav
```
42. Build pitch map and route manually through `voc`:

```bash
pvx pitch-track guide.wav --emit pitch_map --output - | pvx voc target.wav --control-stdin --route pitch_ratio=pitch_ratio --output target_manual_follow.wav
```
43. Modulate stretch from spectral flux:

```bash
pvx pitch-track guide.wav --feature-set all --output - | pvx voc target.wav --control-stdin --route stretch=affine(spectral_flux,0.04,1.0) --route stretch=clip(stretch,0.8,1.6) --output target_flux_stretch.wav
```
44. Modulate pitch from MFCC component:

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 13 --output - | pvx voc target.wav --control-stdin --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --output target_mfcc_pitch.wav
```
45. Gate effect strength by voicing probability:

```bash
pvx pitch-track guide.wav --feature-set all --output - | pvx voc target.wav --control-stdin --route stretch=affine(voicing_prob,0.4,0.8) --output target_voicing_gate.wav
```
46. Route onset strength into dynamic stretch pull:

```bash
pvx pitch-track guide.wav --feature-set all --output - | pvx voc target.wav --control-stdin --route stretch=affine(onset_strength,0.2,0.9) --route stretch=clip(stretch,0.9,1.3) --output target_onset_push.wav
```
47. Use MPEG-7 spectral descriptors for wild macro motion:

```bash
pvx pitch-track guide.wav --feature-set all --output - | pvx voc target.wav --control-stdin --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --output target_mpeg_flux.wav
```
48. Build a "rhythm-following pad" from spoken guide:

```bash
pvx follow speech.wav pad.wav --feature-set all --emit pitch_map --route stretch=affine(transientness,0.2,0.9) --output pad_rhythm_follow.wav
```
49. Use low-confidence pitch regions as chaos zones:

```bash
pvx pitch-track guide.wav --feature-set all --output - | pvx voc target.wav --control-stdin --route stretch=affine(confidence,-0.5,1.5) --route stretch=clip(stretch,0.6,2.0) --output target_conf_chaos.wav
```
50. Create sidechain-driven timing from pitch map with manual routing:

```bash
pvx pitch-track guide.wav --emit pitch_map --output - | pvx voc target.wav --control-stdin --route stretch=pitch_ratio --route pitch_ratio=const(1.0) --output target_pitchtime_follow.wav
```

## 51-60: Control-Map Authoring With Envelope + Reshape

51. Generate ADSR stretch macro map:

```bash
pvx envelope --mode adsr --duration 12 --rate 20 --attack-sec 0.3 --decay-sec 1.0 --sustain 1.4 --release-sec 2.0 --key stretch --output controls/stretch_adsr.csv
```
52. Generate sine pitch-ratio wobble map:

```bash
pvx envelope --mode sine --duration 10 --rate 40 --start 1.0 --peak 0.08 --sine-cycles 12 --key pitch_ratio --output controls/pitch_wobble.csv
```
53. Generate exponential ramp map for acceleration:

```bash
pvx envelope --mode exp --duration 8 --rate 30 --start 0.7 --end 1.8 --exp-curve 6 --key stretch --output controls/stretch_exp.csv
```
54. Normalize a control map to safe range:

```bash
pvx reshape controls/stretch_raw.csv --key stretch --operation normalize --target-min 0.8 --target-max 1.5 --output controls/stretch_norm.csv
```
55. Resample map to dense control rate:

```bash
pvx reshape controls/stretch_env.csv --key stretch --operation resample --rate 80 --interp polynomial --order 5 --output controls/stretch_dense.csv
```
56. Smooth noisy map before applying:

```bash
pvx reshape controls/stretch_noisy.csv --key stretch --operation smooth --window 11 --output controls/stretch_smooth.csv
```
57. Time-shift map late by half second:

```bash
pvx reshape controls/stretch_env.csv --key stretch --operation time-shift --offset 0.5 --output controls/stretch_shifted.csv
```
58. Time-scale map to half speed modulation:

```bash
pvx reshape controls/stretch_env.csv --key stretch --operation time-scale --factor 2.0 --output controls/stretch_slowmod.csv
```
59. Invert a modulation map for opposite behavior:

```bash
pvx reshape controls/mix.csv --key response_mix --operation invert --output controls/mix_inverted.csv
```
60. Build map then apply it in two deterministic steps:

```bash
pvx envelope --mode adsr --duration 8 --rate 20 --key stretch --output controls/stretch_env.csv && pvx reshape controls/stretch_env.csv --key stretch --operation resample --rate 50 --interp polynomial --order 5 --output controls/stretch_env_dense.csv && pvx voc input.wav --stretch controls/stretch_env_dense.csv --interp linear --output input_envdriven.wav
```

## 61-70: Analysis/Response Artifact and Spectral Operator Ideas

61. Cache PV analysis for repeated experiments:

```bash
pvx analysis create source.wav --output source.pvxan.npz --n-fft 4096 --hop-size 256
```
62. Inspect artifact stats for pipeline sanity:

```bash
pvx analysis inspect source.pvxan.npz --json
```
63. Build median response profile:

```bash
pvx response create source.pvxan.npz --method median --normalize peak --output source.pvxrf.npz
```
64. Build RMS response profile for denser weighting:

```bash
pvx response create source.pvxan.npz --method rms --normalize rms --output source_rms.pvxrf.npz
```
65. Apply static spectral imprint:

```bash
pvx filter target.wav --response source.pvxrf.npz --response-mix 1.0 --output target_imprint.wav
```
66. Animate response imprint over time:

```bash
pvx tvfilter target.wav --response source.pvxrf.npz --tv-map controls/response_mix.csv --tv-interp linear --output target_tv_imprint.wav
```
67. Use response-referenced noise filtering:

```bash
pvx noisefilter noisy.wav --response noise_profile.pvxrf.npz --noise-floor 1.2 --output noisy_cleaner.wav
```
68. Exaggerate dominant response peaks:

```bash
pvx bandamp source.wav --response source.pvxrf.npz --band-gain-db 10 --peak-count 12 --output source_peakpump.wav
```
69. Compress/expand in response-relative spectral space:

```bash
pvx spec-compander source.wav --response source.pvxrf.npz --comp-threshold-db -22 --comp-ratio 3 --expand-ratio 1.4 --output source_specdyn.wav
```
70. Pitch-shift response curve instead of source pitch:

```bash
pvx filter source.wav --response source.pvxrf.npz --transpose-semitones 7 --shift-bins 24 --output source_resp_shifted.wav
```

## 71-80: Ring, Resonator, Chordmapper, Inharmonator

71. Slow ring tremolo with deep mix:

```bash
pvx ring input.wav --frequency-hz 3.5 --depth 1.0 --mix 1.0 --output input_ring_trem.wav
```
72. Audio-rate ring metallicizer:

```bash
pvx ring input.wav --frequency-hz 97 --depth 0.95 --mix 0.9 --output input_ring_metal.wav
```
73. Ring with near-chaotic feedback memory:

```bash
pvx ring input.wav --frequency-hz 43 --depth 1.0 --mix 1.0 --feedback 0.92 --output input_ring_fb.wav
```
74. Ring + resonant peak filter:

```bash
pvx ringfilter input.wav --frequency-hz 60 --resonance-hz 1600 --resonance-q 12 --resonance-mix 0.6 --output input_ringfilter.wav
```
75. Time-varying ring map performance:

```bash
pvx ringtvfilter input.wav --tv-map controls/ring_map.csv --tv-interp cubic --output input_ringtv.wav
```
76. Chord-lock noisy ambience to minor harmony:

```bash
pvx chordmapper ambience.wav --root-hz 110 --chord minor --strength 1.0 --boost-db 8 --attenuation 0.3 --output ambience_minorlock.wav
```
77. Tight chord attraction with low tolerance:

```bash
pvx chordmapper pad.wav --root-hz 220 --chord diminished --tolerance-cents 12 --output pad_tight_chord.wav
```
78. Soft chord attraction with high tolerance:

```bash
pvx chordmapper pad.wav --root-hz 220 --chord major7 --tolerance-cents 80 --output pad_soft_chord.wav
```
79. Inharmonic stiff-string emulation:

```bash
pvx inharmonator bells.wav --inharmonic-f0-hz 220 --inharmonicity 0.0002 --inharmonic-mix 1.0 --output bells_stiff.wav
```
80. Subtle inharmonic shimmer blend:

```bash
pvx inharmonator pad.wav --inharmonic-f0-hz 110 --inharmonicity 0.00005 --inharmonic-mix 0.25 --dry-mix 0.8 --output pad_shimmer.wav
```

## 81-90: Multi-Stage Pipelines, Streaming, and Batch Tricks

81. One-line cleanup + stretch + formant chain:

```bash
pvx chain vocal.wav --pipeline "denoise --reduction-db 6 | deverb --strength 0.35 | voc --stretch 1.3 | formant --mode preserve" --output vocal_full_chain.wav
```
82. Stream with explicit output policy controls:

```bash
pvx stream source.wav --output source_stream.wav --chunk-seconds 0.2 --time-stretch 2.0 --bit-depth 24 --dither tpdf --dither-seed 7 --metadata-policy sidecar
```
83. Stream wrapper compatibility baseline:

```bash
pvx stream source.wav --mode wrapper --output source_wrapper.wav --chunk-seconds 0.2 --time-stretch 2.0
```
84. Stateful stream driven by stretch map:

```bash
pvx stream input.wav --output output_stream_map.wav --chunk-seconds 0.2 --stretch controls/stretch.csv --interp linear
```
85. Pipe denoise straight into deverb with no temp file:

```bash
pvx denoise field.wav --reduction-db 5 --smooth 7 --stdout | pvx deverb - --strength 0.35 --output field_clean.wav
```
86. Pipe harmonize into voc for post-stack stretch:

```bash
pvx harmonize lead.wav --intervals 0,4,7 --gains 1,0.75,0.65 --stdout | pvx voc - --stretch 1.5 --output lead_stack_stretched.wav
```
87. Chain with alternating transient modes for A/B blend:

```bash
pvx chain speech.wav --pipeline "voc --transient-mode hybrid --stretch 1.5 | voc --transient-mode off --stretch 1.0" --output speech_transient_combo.wav
```
88. Run two-pass pitch: retune then ratio-jump:

```bash
pvx chain vocal.wav --pipeline "retune --root E --scale minor --strength 0.7 | voc --ratio 9/8" --output vocal_two_pass_pitch.wav
```
89. Long-run resume-safe rendering script target:

```bash
pvx voc long.wav --target-duration 14400 --checkpoint-dir ckpt_long --auto-segment-seconds 0.25 --resume --output long_4h.wav
```
90. Dynamic spectral map workflow with reusable artifacts:

```bash
pvx analysis create source.wav --output source.pvxan.npz && pvx response create source.pvxan.npz --output source.pvxrf.npz && pvx tvfilter source.wav --response source.pvxrf.npz --tv-map controls/mix.csv --output source_dynspec.wav
```

## 91-100: QA, Stress, and Reproducibility Challenges

91. Budget-gate every render before execution:

```bash
pvx stretch-budget input.wav --disk-budget 10GB --requested-stretch 5000 --fail-if-exceeds
```
92. Compare quality at multiple FFT sizes quickly:

```bash
pvx chain source.wav --pipeline "voc --stretch 2 --n-fft 1024 | voc --stretch 1 --n-fft 4096" --output source_fft_compare.wav
```
93. Build deterministic dithered exports for null tests:

```bash
pvx voc input.wav --stretch 1.1 --bit-depth 24 --dither tpdf --dither-seed 123 --output out_seeded.wav
```
94. Generate manifest metadata for every run:

```bash
pvx voc input.wav --stretch 1.08 --manifest-json reports/pvx_runs.json --manifest-append --output out_manifest.wav
```
95. Stress dynamic controls with dense maps:

```bash
pvx voc input.wav --stretch controls/stretch_dense.csv --interp polynomial --order 5 --output out_densectrl.wav
```
96. Test phase behavior by toggling phase engines:

```bash
pvx chain source.wav --pipeline "voc --stretch 2 --phase-engine propagate | voc --stretch 1 --phase-engine random" --output source_phase_duel.wav
```
97. Validate sidechain routing with built-in examples:

```bash
pvx follow --example all
```
98. Run parity benchmark regression module directly:

```bash
python3 -m unittest tests.test_pvc_parity_benchmark
```
99. Run full regression tests after extreme experiments:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
100. Build a curated "crazy pack" by rendering 100 variants and scoring them with metrics:

```bash
pvx voc input.wav --stretch 1.2 --output candidate.wav && python3 -m unittest tests.test_cli_regression
```

---

Tips:
- Keep experimental commands in scripts with fixed seeds/options for repeatability.
- For very long jobs, always use `--checkpoint-dir` and `--resume`.
- Validate impossible requests with `pvx stretch-budget` before running expensive renders.
