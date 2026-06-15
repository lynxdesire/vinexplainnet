# Project Context

    project_name       : vinexplainnet                                  [HIGH]
    domain             : consumer-grade OCT retinal imaging —           [HIGH]
                         multi-task disease classification + attention
                         explainability + edge deployment
    framework          : PyTorch 2.x (plain torch.nn; hand-built        [HIGH]
                         depthwise-separable EfficientNet-B0 backbone)
    venue              : IEEE Transactions on Consumer Electronics      [HIGH]
    primary_datasets   : 2 datasets (Kermany/UCSD, Duke)                [HIGH]
    compute_target     : 4x NVIDIA A100 80GB, FP16, ~36 h              [HIGH]
    hparams_reference  : Appendix A, Table XV                           [HIGH]
    supp_path          : none (Appendices A-E are in-body)
    extra_signals      : 1 algorithm box; no released weights;
                         external Stage-1 pretraining; n=23 clinician
                         study (human, surrogate ratings for tests)

    NEEDS_USER_DECISION: 0
    Distinctness axes (layout / config-CLI / README) fixed by the team.

## 1. project_name

`vinexplainnet`. The method is named VINExplainNet (Vascular–Immune–Neural
Explainable Network). Content words from the title with stopwords removed.

## 2. supp_path

`none`. No separate supplementary file ships with the manuscript; Appendices
A–E (hyperparameters, explanation visualisation, confusion matrix, algorithmic
details) are printed inside the main text. Source location for this package:
manuscript Reproducibility Statement + Appendix A.

## 3. domain

Consumer-grade optical coherence tomography (OCT) retinal-health monitoring.
Concretely: B-scan multi-task disease classification with inherently
interpretable attention explanations, optimised for edge hardware. Not a
generic "medical imaging" project — every design choice targets noisy,
motion-degraded consumer OCT and sub-200 ms on-device latency.
Source: Abstract; §I Introduction; Index Terms.

## 4. framework

PyTorch 2.x, plain `torch.nn`. The paper states the model "has been created in
PyTorch 2.0" (§IV-C Implementation Details). The EfficientNet-B0 backbone is
re-expressed with depthwise-separable convolutions inline (no `timm`,
no torchvision model zoo) so the DWS-Conv channel pruning and the ASS input
layer can be implemented exactly. ONNX export targets `torch.onnx` with
`dynamo=False`.
Source: §IV-C; §III-B-1.

## 5. venue

IEEE Transactions on Consumer Electronics (TCE). Evidence: Index Terms list
"consumer electronics"; the reference list uses the IEEE Trans. Consum.
Electron. abbreviation and IEEE two-column citation style; the manuscript
framing (edge devices, energy per inference, consumer wellness ecosystem) is
characteristic of TCE.
Source: Index Terms; reference formatting; §VI Discussion.

## 6. primary_datasets

| name | version / split | classes | size | access |
| --- | --- | --- | --- | --- |
| Kermany (UCSD) | train 83,484 / test 1,000 (250 per class) | CNV, DME, Drusen, Normal | 108,312 B-scans, 496×512, Spectralis | https://data.mendeley.com/datasets/rscbjbr9sj/3 (CC BY 4.0) |
| Duke (Srinivasan) | 5-fold cross-validation | AMD, DME, Normal | 3,231 B-scans, 45 patients, 496×768, Spectralis SD-OCT | https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm |

Both are public. Real bytes are large and licensing/registration varies, so
the package ships a deterministic synthetic layered-retina OCT generator with
per-class recoverable lesion signatures plus a manifest loader for the real
path. Simulated consumer-imaging degradation (motion blur, 2–3× speckle,
50–75% resolution reduction, ±30% brightness/contrast) is applied per §IV-A-4.
Source: §IV-A; dataset footnotes.

## 7. compute_target

4× NVIDIA A100 (80 GB) GPUs, mixed precision FP16, ~36 h wall-clock for the full
three-stage procedure (Appendix A). Reported edge-inference targets:
Jetson Nano 34 ms / Snapdragon 888 19 ms (optimised), model size 3.2 MB,
energy 28 mJ per inference (Table VII, XII, XIII).
Source: §IV-C; Appendix A; Tables VII/XII/XIII.

## 8. hparams_reference

Appendix A, Table XV (Training Hyperparameters):

| parameter | value |
| --- | --- |
| input resolution | 224 × 224 |
| batch size | 64 |
| optimizer | AdamW |
| initial lr | 1e-4 |
| minimum lr | 1e-6 |
| weight decay | 1e-4 |
| epochs (Stage 2) | 200 |
| epochs (Stage 3) | 50 |
| lr schedule | cosine annealing |
| λ_cls | 1.0 |
| λ_e | 0.1 |
| λ_r | 5e-4 |
| attention dim d_k | 80 |
| ASS σ_s | 10.0 |
| ASS σ_r | 0.1 |
| pruning threshold τ | 0.3 × max(I_k) |
| quantization bit-width | 8-bit (first conv + final classifier kept FP16) |

Scattered prose values: backbone channels (24, 40, 112, 320); MLP hidden 128;
Stage-1 pretrain 100 ImageNet epochs + 50 medical-imaging (CheXpert) epochs;
bootstrap CI with 1000 iterations; McNemar with Bonferroni at α = 0.01.

## 9. extra_signals

- Algorithm 1 (VINExplainNet training procedure) — 1 algorithm box.
- No released checkpoints; weights "available with the final published version".
- Stage-1 backbone pretraining uses external ImageNet-1K + a large medical-imaging
  corpus (CheXpert). External-corpus pretraining is out of scope for the smoke
  path; the stage hook is implemented and a manifest path is accepted
  (logged as a deviation).
- Clinical study with n = 23 board-certified ophthalmologists (Likert ratings,
  Tables IV/V). Human judgements are not code-reproducible; the evaluation module
  computes the Likert summary statistics from a ratings file and a deterministic
  synthetic ratings generator backs the tests.
- Faithfulness is a deletion-correlation metric (Eq. 26) — fully implementable.
- Code-availability statement, verbatim from the Reproducibility Statement:
  "Model architecture set up and training scripts will be posted with the final
  published versions of this article, as allowed by journal regulations."

## Headline numbers (for the reviewer-facing reproduction targets)

- Kermany: ACC 98.47%, AUC 99.62%, 4.2M params.
- Duke: ACC 97.63%.
- Explanation: faithfulness 0.912, Gini sparsity 0.687, clinical alignment 4.23/5.
- HAFM gain over backbone: +1.92% (96.20 → 98.12). ASS gain: +0.22% (+2.1% under high noise).
- Edge (optimised): Jetson Nano 34 ms, Snapdragon 888 19 ms, 3.2 MB, 28 mJ.
- Cross-dataset generalisation mean: 85.64%.
