# Implementation Map

This file is the single home of paper provenance: source files carry no
comments and no docstrings, so every equation, algorithm box, table and metric
is traced here. Columns: paper location | item | file | symbol | notes.

## Architecture (§III-B)

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| §III-B-1, Eq. 2 | EfficientNet-B0 DWS-Conv backbone, hierarchical feats F1..F4 (24,40,112,320) | `vinexplainnet/lens/backbone.py` | `EfficientBackbone` | depthwise-separable stages; returns 4-level feature pyramid |
| §III-A, Eq. 3 | Adaptive Speckle Suppression, x' = (1-α)x + α·Bilateral(x,σs,σr), α=σ(α̃) | `vinexplainnet/lens/speckle.py` | `AdaptiveSpeckleSuppression` | learnable scalar α via sigmoid; differentiable bilateral approx |
| §III-B-2, Eq. 4 | local attention A_local = σ(Conv1×1(DWConv3×3(F4))) | `vinexplainnet/lighting/streams.py` | `LocalStream` | single-channel spatial map at F4 resolution |
| §III-B-2, Eq. 5 | regional attention A_regional = σ(Conv1×1(DilConv_{r=2}(F3))) | `vinexplainnet/lighting/streams.py` | `RegionalStream` | dilated conv rate 2 at F3 resolution |
| §III-B-2, Eq. 6-9 | global self-attention on F2, dk=80, A_global = σ(Reshape(O·W_O)) | `vinexplainnet/lighting/streams.py` | `GlobalStream` | lightweight Q/K/V on flattened F2; scaled dot-product |
| §III-B-2, Eq. 10 | fusion A_fused = wl·Â_l + wr·Â_r + wg·Â_g, softmax(w) | `vinexplainnet/lighting/fusion.py` | `HierarchicalAttentionFusion` | upsample-align to F4; learnable weights normalised by softmax |
| §III-B-2, Eq. 11 | F_att = F4 ⊙ Expand(A_fused) + F4 | `vinexplainnet/lighting/fusion.py` | `HierarchicalAttentionFusion.modulate` | channel-broadcast residual modulation |
| §III-B-3, Eq. 12-13 | classification head GAP + 2-layer MLP (hidden 128), sigmoid | `vinexplainnet/framing/heads.py` | `PredictionHead` | per-dataset class count C |
| §III-B-4, Eq. 14-15 | class explanation E_c = ReLU(Σ α_k^c · A_fused^(k)), α^c = softmax(W^exp·h) | `vinexplainnet/captions/generator.py` | `ExplanationGenerator` | class-specific projection over fused-map channels |
| §III-B-4, Eq. 16 | refined E_c = Conv1×1(E_c ∥ F1_up) | `vinexplainnet/captions/generator.py` | `ExplanationGenerator.refine` | concat upsampled low-level features |

## Losses (§III-C, Eq. 1, 17-22)

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Eq. 1 | L_total = Σ_d λ_d L_cls + λ_e L_explain + λ_r L_reg | `vinexplainnet/notes/objective.py` | `TotalObjective` | aggregates the terms with Table XV weights |
| Eq. 17 | weighted BCE classification loss (class-balanced w+/w-) | `vinexplainnet/notes/classification.py` | `weighted_bce` | inverse-frequency weights |
| Eq. 18-19 | L_coherence = Σ ‖E(i,j)-E(i+1,j)‖² + ‖E(i,j)-E(i,j+1)‖² | `vinexplainnet/notes/explanation.py` | `coherence_term` | total-variation style anisotropic smoothness |
| Eq. 20 | L_sparsity = ‖E‖_1 / (H'·W') | `vinexplainnet/notes/explanation.py` | `sparsity_term` | mean absolute activation |
| Eq. 21 | L_boundary = ‖∇E - β·∇x‖² | `vinexplainnet/notes/explanation.py` | `boundary_term` | Sobel gradients align explanation to anatomy |
| Eq. 22 | L_reg = Σ ‖θ‖² | `vinexplainnet/notes/regularization.py` | `l2_penalty` | over weight tensors |

## Training (§III-D, Eq. 23, Algorithm 1)

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Algorithm 1 | full training procedure (augment → backbone → HAFM → fuse → predict → explain → AdamW step → cosine decay) | `vinexplainnet/direction/runner.py` | `Director.shoot` | one method per act |
| §III-D-1 | three stages: backbone pretrain / task-specific / explanation refinement | `vinexplainnet/direction/stages.py` | `Stage`, `STAGE_PLAN` | Stage 1 hook external (deviation D2) |
| Eq. 23 | cosine annealing η_t = η_min + ½(η0-η_min)(1+cos(πt/T)) | `vinexplainnet/direction/schedule.py` | `cosine_lr` | T=200 (Stage 2) |
| §III-D-2 | data augmentation (geometric / intensity / noise / OCT-specific) | `vinexplainnet/casting/augment.py` | `ConsumerAugment` | per the bullet list |
| §III-D | AdamW optimiser, atomic checkpoints | `vinexplainnet/direction/clapper.py` | `make_optimizer`, `save_take` | tmp file + os.replace |

## Edge optimisation (§III-E, Eq. 24-25)

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Eq. 24 | channel importance I_k = (1/N)Σ |∂L/∂F_k|·‖F_k‖ | `vinexplainnet/cutting/prune.py` | `channel_importance` | gradient×activation score |
| §III-E-1 | structured channel pruning at τ = 0.3·max(I_k) | `vinexplainnet/cutting/prune.py` | `structured_prune` | drops sub-threshold channels |
| Eq. 25 | post-training int8 quantization, s = (Wmax-Wmin)/(2^b-1) | `vinexplainnet/cutting/quantize.py` | `quantize_tensor`, `MixedPrecisionQuantizer` | first conv + final classifier kept FP16 |

## Data (§IV-A)

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| §IV-A | synthetic layered-retina OCT generator + per-class lesion signatures | `vinexplainnet/casting/synthetic.py` | `SyntheticOCT` | recoverable CNV/DME/Drusen signal |
| §IV-A-1/2 | Kermany/Duke manifest adapters + class maps | `vinexplainnet/casting/sources.py` | `ManifestSource`, `DATASET_SPECS` | real-data path |
| §IV-A-4 | simulated consumer degradation (blur, speckle, resolution, contrast) | `vinexplainnet/casting/degrade.py` | `ConsumerDegradation` | applied at evaluation |
| §IV-A | batching, splits | `vinexplainnet/casting/reels.py` | `Reel`, `make_loaders` | plain numpy/torch loader |

## Evaluation (§IV-B, Eq. 26; §V Tables I-XIV)

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| §IV-B-1 | ACC/AUC/SEN/SPE/F1 macro (Tables I-III, IX) | `vinexplainnet/screening/metrics.py` | `classification_report` | macro-averaged |
| Eq. 26 | faithfulness deletion-correlation (Table IV/VIII) | `vinexplainnet/screening/faithfulness.py` | `deletion_faithfulness` | masking-order vs prediction-drop correlation |
| §IV-B-2 | Gini sparsity of explanation maps (Table IV) | `vinexplainnet/screening/faithfulness.py` | `gini_sparsity` | concentration of attention |
| §IV-B-2 | clinical Likert summary (Tables IV/V) | `vinexplainnet/screening/clinical.py` | `clinical_alignment` | mean + bootstrap CI; synthetic ratings for tests |
| §V-I, Table XI | McNemar test + Bonferroni correction | `vinexplainnet/screening/significance.py` | `mcnemar_test`, `bonferroni` | α = 0.01 |
| §V-I | bootstrap 95% CI (1000 iters) | `vinexplainnet/screening/significance.py` | `bootstrap_ci` | accuracy CI |
| Table VI | robustness under simulated degradation | `vinexplainnet/screening/robustness.py` | `degradation_sweep` | relative-drop table |
| Table VII/XII/XIII | edge latency / size / GFLOPs / energy / memory | `vinexplainnet/screening/footprint.py` | `profile_footprint` | param/FLOP counting + latency timer |
| Table X | cross-dataset generalisation | `vinexplainnet/screening/transfer.py` | `cross_dataset` | no fine-tuning on target |

## Assembly & interface

| paper | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| §III-B overview | assembled VINExplainNet model | `vinexplainnet/feature.py` | `VINExplainNet`, `build_feature` | wires lens→lighting→framing→captions |
| §IV-C | config schema + HOCON loader | `vinexplainnet/treatment/schema.py`, `loader.py` | `Treatment`, `read_treatment` | frozen dataclasses |
| — | CLI verbs train/evaluate/explain/optimize/export | `vinexplainnet/studio/clapperboard.py` | argh commands | `python -m vinexplainnet.studio` |
| — | seeding / logging / io / device / types | `vinexplainnet/crew/*.py` | — | single `set_seed`; atomic io |

## Deviations
See `docs/deviations.md`.
