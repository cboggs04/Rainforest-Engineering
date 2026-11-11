# Phase 1-3 audit (reproduced with scripts in this folder)

Run: `python experiments/audit/{baseline_eval,preproc_and_int8,corruption_test}.py` from repo root (folds are regenerated deterministically).
Reusable code: `src/audio_preprocessing.py`, `src/evaluation.py`. No existing checkpoint/header was modified.

## Artifacts found
- Models: BN09 5-fold base (`..._3Class_BN09_base_fold{1-5}`), `..._3Class_base_fold1`, `Augmented_V1`, `Finetuned_V1`, `Finetuned_V2`, `ESP32_Deployment_V1/*INT8.tflite` + mel filterbank .npy.
- Data: FSC22 wav44 + csv, `audios_5_20` (1950 x 100000 @20k), `mel_features_5_20_fold1-5`, V1/V2 augmented wavs+mel+metadata CSVs.
- NOT in repo: V2 training notebook state is only in the notebook; `rainforest_model.h`/`mel_filterbanks.h`/.ino live in `~/Documents/Arduino/Rainforest_Detection_ESP32/`, not in git. `requirements.txt` absent.
- `esp32_recording.wav` is 100% zeros (failed capture) -> unusable.

## Reproduced baselines (Keras, fold-5 clean val n=165 [135/15/15]; noisy val n=22 [11 cs / 11 gs / 0 bg])
| Model | clean acc | clean bal-acc | cs / gs recall (clean) | noisy acc | cs / gs recall (noisy) |
|---|---|---|---|---|---|
| Original (BN09 fold5) | 91.5% | 0.788 | 0.667 / 0.733 | 63.6% | 0.636 / 0.636 |
| V1 | 86.7% | 0.728 | 0.600 / 0.667 | 72.7% | 0.818 / 0.636 |
| V2 | 90.3% | 0.763 | 0.600 / 0.733 | 54.5% | 0.636 / 0.455 |
All historical numbers in the handoff reproduce exactly.
INT8 V1 on clean val: acc 86.7%, bal-acc 0.768, Keras/INT8 agreement 97.0% (5 disagreements, all low-confidence), max |dp| 0.356.

## Findings
1. **Evaluation is statistically too weak to rank models.** Noisy val = 22 clips (1 clip = 4.5 pp), 17 unique sources, no background class. Clean val has 15 chainsaw + 15 gunshot. V1>Original rests on 2 clips and V1 is *worse* on clean val (bal-acc 0.728 vs 0.788).
2. **Leakage.** Clean folds are random `StratifiedKFold` over clips (FSC22's official `fold` column is ignored); noisy mixes use the official folds. 6/17 noisy-val threat sources are also in V1's clean training set (pickle folds 2-4). Freesound IDs in filenames are unique per clip, so source-recording grouping cannot be recovered from metadata.
3. **Training data is tiny**: 75 chainsaw + 75 gunshot clips total, trained from scratch (`weights=None`). 17 of 26 FSC22 classes (speech, firework, generator, handsaw, axe, tree-falling, etc.) are discarded rather than used as hard negatives.
4. **Firmware padding bug**: librosa 0.11 `melspectrogram` default pad is `constant` (zeros); firmware reflect-pads. Using zero padding in the emulator gives 0.0000 dB difference vs training features; reflect gives up to 2.75 dB mean error in frame 0 / 1.5 dB in last frame (98.8% prediction agreement). Fix is a one-line firmware change; no retrain.
5. **Level sensitivity** (INT8, clean val target clips): chainsaw recall 0.73 @x1 -> 0.40 @x10, 0.20 @x0.1; chainsaw->gunshot 7% -> 33% @x10; background->gunshot false alarm 5% @x1 -> 22% @x10 -> 45% @x30. Model uses absolute dB level (no per-clip normalisation).
6. **Probable cause of live chainsaw->GUNSHOT 97%**: the live capture had 27,940 exact zeros (28%). Simulating 28% zeroed 512-sample blocks (= firmware `BLOCK_SAMPLES`) on all 75 chainsaw clips -> 79% classified GUNSHOT (mean P 0.77). Random single-sample zeros, DC offset, x2/x0.5 gain, clipping, 18-bit quantisation do NOT cause this. Strong hypothesis: I2S read dropouts, not a model-only error. UNCONFIRMED until a real ESP32 capture is inspected (are zero runs 512-aligned?). The firmware health check only rejects all-zero audio.

## Proposed smallest next steps
a. Get a real ESP32 capture (fix/debug `record_esp32_audio.py` path; current file is all zeros) and check zero-run structure, then run through `reference_mel` + INT8.
b. Firmware: zero-pad instead of reflect; reject recordings with >1% zeros or any zero run >=64 samples; remove DC.
c. Then, before any V3: rebuild eval with official-fold grouping, far more mixed clips (hundreds, multiple SNR/position), and FSC22 hard negatives.

## Phase 4: leak-free evaluation set (datasets/eval_v1, built by experiments/eval_set/build_eval.py)
Official FSC22 fold 5 only: 135 bg + 225 hard-negative (15 non-target classes, must not alert) + clean/gain-shifted cs/gs +
600 mixes (5 SNRs x 4 random bg/position per source clip; achieved event-window SNR asserted). 95% CIs bootstrap over the 15 source clips.
Scores: experiments/eval_set/baseline_scores.csv.
CAVEAT: Original/V1/V2 were trained on ~80% of fold-5 clips (random pickle folds), so clean recall (1.00) is inflated; only
retrained models give an honest number. Relative mixed/hard-neg/gain results are still informative:
- Gunshot-in-background is the dominant failure: recall 0.10-0.57 even at +15 dB event SNR (V1 best: 0.22 @-5dB ... 0.57 @+15dB).
- Hard-negative false alarm 21-29% for every model (never trained on those classes).
- Gain x0.3/x3 drops chainsaw recall 1.00 -> 0.67-0.87.
- V1 > Original on mixes (cs +5dB: 0.92 vs 0.80; gs +10dB 0.43 vs 0.25); V2 not better than V1. INT8 ~= Keras.

## V3-b (experiments/v3/train_b.py; 3 seeds each; train FSC22 folds 1-3 + external train sources)
External data: ESC-50 (40 chainsaw, hard negatives, bg) + UrbanSound8K (374 gunshot, mechanical hard negatives); both CC BY-NC.
31 gunshot + 2 chainsaw external clips were near-verbatim FSC22 copies (NCC>0.8) -> whole source recordings excluded. Splits grouped by Freesound ID.
Second test set datasets/eval_ext = held-out external sources (66 gs / 13 cs / 551 hard-neg / 109 bg clips).
B0 = +external data, balanced mixes; B1 = B0 + non-target events mixed into rainforest noise (label 0); B2 = B1 + ImageNet init.
Mean over 3 seeds (eval_ext | eval_v1):
- bg false alarm: B0 .25|.13  B1 .18|.10  B2 .06|.05  (V1 .22 on ext)
- hard-neg false alarm: B0 .17|.24  B1 .16|.24  B2 .05|.13 (V1 .39 on ext)
- gunshot in noise +5dB: B0 .75|.65  B1 .74|.61  B2 .90|.66 (V1 .23 on ext)
- chainsaw in noise -5dB: B0 .82|.58  B1 .71|.56  B2 .82|.73
- chainsaw->gunshot: B2 .01|.01
B2 is best on nearly every metric with the lowest seed variance (val bal-acc .851-.864). Remaining weakness: FSC22 gunshots in noise (.56 @-5dB).
Candidate for deployment: B2 seed 0 (highest val bal-acc, chosen on validation not test): models/MobileNetV3_Small_Rainforest_V3b_B2_s0/best.keras

## V3-c (cfg C = B2 + gain down to x0.03 + simulated speaker/room/mic chain on 30% of examples + 450 loud-bed mixes; 3 seeds)
New stress set datasets/eval_stress (test sources only): simulated replay, quiet (x0.03/x0.1), loud beds.
- Gains vs B2: quiet gunshot .82->.95, quiet chainsaw .81->.96, simulated-replay gunshot .86->.94; real phone->Mac gunshots 1-4/7 (B2 0-2/7, 7 clips, labels unconfirmed).
- Costs: false alarms ~double at threshold 0.5 (ext bg .06->.13, hard-neg .05->.09, FSC hard-neg .13->.17); loud-bed mixes did not help.
- At matched false-alarm rate (C@0.8 vs B2@0.5) C is better on replay/quiet but worse on gunshots in noise (ext +5dB .78 vs .90, -5dB .41 vs .58). Neither dominates.
- Simulation is too gentle vs real phone playback -> collect real replay data with record_replay_dataset.py (playlists in replay_playlists/).

## V3-d (D = V3-c fine-tuned with real phone->Mac replay train session, 3 seeds) and INT8 attempt
Real replay TEST session (58 clips, unseen sources), mean 3 seeds @0.5: B2 gs .50 cs .70 FA .04 | C gs .75 cs .92 FA .14 | D gs .78 cs .86 FA .08.
D keeps C's quiet/replay gains with false alarms near B2 (ext bg .08, hard-neg .06). Candidate float model: V3b_D_s1 (best val).
INT8 of D_s1 FAILED validation (files renamed *_REJECTED_*): float/INT8 agreement 72-79% (V1 was 97%), INT8 recall collapses
(ext gs +5dB .86->.57, replay gs .75->.55). Ops/size are fine (same 8 ops, 1.19 MiB).
Cause isolated by converting one checkpoint per stage (400 ext clips, reference kernels): scratch B0 94.8%, B1 97.8% vs pretrained B2 88.2%, C 86.3%, D 81.5%.
Only pretrained models have 4 zero-scale (dead) squeeze-excite tensors and wider activation ranges (8-10 layers with scale>0.5 vs 1).
Option 1 (surgical fix) tried and FAILED:
- Function-preserving patch of dead squeeze-excite channels (experiments/v3/patch_se.py; float max|dp| 3e-6): INT8 agreement 81.5% -> 81.2%. Dead SE blocks are not the cause.
- 16x8 quantization (int16 activations) to test the activation-range hypothesis: converter refuses ("16x8 not yet supported for op HARD_SWISH"), so it is also not a deployment path.
Remaining options: QAT (tfmot needs tf_keras rebuild; MobileNetV3 Multiply/hard-swish need custom quantize configs) or retrain the D recipe from scratch (scratch models quantize at 95-98%).

## Option 3: scratch training (INT8-friendly)
- V3-e (D recipe from scratch, seed 0): INT8 agreement 95-100% but weak chainsaw (ext clean .69) and chainsaw->gunshot 6-11% -> rejected.
- V3-f (same, + distillation from the 3-seed V3-d ensemble, 50/50 hard/soft targets, seed 0, best epoch 25/30):
  INT8 agreement 96.7-100%; INT8 ext: bg FA .10, hard-neg FA .11, chainsaw -5dB .65, clean chainsaw .92, gunshot +5dB .77, cs->gs .04;
  real replay test: gunshot .90 (18/20), chainsaw .58 (7/12), FA .12. Gap vs float V3-d remains on chainsaw in noise/replay.
DEPLOYMENT CANDIDATE: models/ESP32_Deployment_V3/MobileNetV3_Rainforest_V3f_INT8.tflite + rainforest_model_V3f.h
(1,248,680 B; same 8 ops as firmware; 0 zero-scale tensors; input int8 scale 0.558024 zp 51 (firmware reads params at runtime);
reference-kernel agreement: FSC22 96.9%, ext 96.7%, stress 97.1%, real replay 98.3%). Not yet run on the ESP32.
V3-f seeds 1-2 (45-epoch budget; early-stopped at best epochs 4 and 12): no seed dominates on test sets.
Note: per-seed validation sets differ (seeded mixing), so per-seed val bal-acc is not comparable. Re-selected on ONE shared
validation set (validation sources only, seed 777; INT8 reference kernels), priority score = mean(cs,gs recall) - neg FA - 2*cs->gs:
seed0 .522 (FA .151, cs .69, gs .71, cs->gs .015) | seed1 .519 (FA .164, cs .76, gs .68) | seed2 .487. Seeds 0/1 tie within noise;
seed 0 kept (lower false alarms and cs->gs, the stated priorities). Deployment candidate unchanged: rainforest_model_V3f.h (F_s0).
