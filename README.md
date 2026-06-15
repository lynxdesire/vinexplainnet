# VINExplainNet

> PRESS KIT

> A feature presentation in four reels: an OCT B-scan walks in as raw footage
> and walks out as a labelled, captioned diagnosis you can read at the edge.

VINExplainNet (Vascular-Immune-Neural Explainable Network) is a lightweight
multi-task network for consumer optical coherence tomography. It classifies
retinal pathology and, in the same forward pass, draws an attention-based
explanation map that points a clinician at the structures behind each call.

---

## Logline

One small network (4.2M parameters), three attention streams, one honest
explanation per prediction, fast enough for a phone.

## Synopsis

The picture opens on an Adaptive Speckle Suppression layer that learns, through
one sigmoid-bounded dial, how much consumer-grade noise to filter against edge
preservation. A depthwise-separable EfficientNet-B0 backbone produces a
four-level feature pyramid. The Hierarchical Attention Fusion Module (HAFM)
runs three parallel streams over it: a local stream (3x3 depthwise on the
deepest map), a regional stream (dilated convolution on the mid map), and a
global stream (a small self-attention over the shallow map). Their maps are
aligned and combined with softmax-normalised weights, then used both to
modulate features for a two-layer classification head and to compose per-class
visual explanations refined against low-level structure. Training runs in three
acts: backbone pretraining, task-specific multi-task fitting, and an explanation
refinement act that sharpens the captions. The cutting room then prunes channels
by gradient-times-activation importance and quantises to int8, keeping the first
convolution and the final classifier in FP16.

## Cast & Crew

| Department | Module | Role on screen |
| --- | --- | --- |
| Treatment | `vinexplainnet/treatment` | HOCON config schema and loader |
| Casting | `vinexplainnet/casting` | synthetic OCT generator, dataset adapters, degradation, augmentation |
| Lens | `vinexplainnet/lens` | speckle suppression and the DWS-Conv backbone |
| Lighting | `vinexplainnet/lighting` | HAFM attention streams and fusion |
| Framing | `vinexplainnet/framing` | classification head |
| Captions | `vinexplainnet/captions` | per-class explanation generator |
| Notes | `vinexplainnet/notes` | classification, explanation-quality, and weight losses |
| Cutting | `vinexplainnet/cutting` | structured pruning and mixed-precision quantization |
| Direction | `vinexplainnet/direction` | three-act trainer, schedule, checkpoints |
| Screening | `vinexplainnet/screening` | metrics, faithfulness, significance, footprint |
| Crew | `vinexplainnet/crew` | seeding, logging, atomic IO, device |
| Studio | `vinexplainnet/studio` | the `argh` command line |

The assembled feature lives in `vinexplainnet/feature.py`.

## Distribution

pip:

```bash
pip install -r requirements.txt
pip install --no-deps -e .
```

conda:

```bash
conda env create -f environment.yml
conda activate vinexplainnet
pip install --no-deps -e .
```

docker:

```bash
docker build -t vinexplainnet:latest .
docker run --rm vinexplainnet:latest --help
```

## Running order

```bash
python -m vinexplainnet.studio train    --treatment configs/experiment/main.conf
python -m vinexplainnet.studio evaluate --treatment configs/experiment/main.conf
python -m vinexplainnet.studio explain  --treatment configs/experiment/main.conf
python -m vinexplainnet.studio optimize --treatment configs/experiment/main.conf
python -m vinexplainnet.studio export   --treatment configs/experiment/main.conf
```

Swap the treatment for `supplementary_duke.conf` or any `ablation_*.conf`. The
two-step screening short cut is `make smoke`.

## Box office and reviews

The numbers below are the manuscript's targets on the real Kermany / Duke
data. This release ships a synthetic OCT path by default (see Advisories),
so out of the box the commands run end to end but do not land on these figures;
point a treatment's `data.manifest` at downloaded data to chase them.

| Reel | Command | Manuscript figure |
| --- | --- | --- |
| Classification (Kermany) | `evaluate --treatment configs/experiment/main.conf` | ACC 98.47%, AUC 0.9962 |
| Classification (Duke) | `evaluate --treatment configs/experiment/supplementary_duke.conf` | ACC 97.63% |
| Explanation quality | `explain --treatment configs/experiment/main.conf` | faithfulness 0.912, Gini 0.687, clinical 4.23/5 |
| HAFM ablation | `evaluate --treatment configs/experiment/ablation_no_hafm.conf` | +1.92% from baseline to full |
| Edge footprint | `optimize --treatment configs/experiment/main.conf` | 3.1M params, 3.2 MB, Jetson 34 ms |

## Technical specs

| Item | Value |
| --- | --- |
| Training rig | 4x NVIDIA A100 80GB, FP16 |
| Wall-clock | ~36 h for the three-act schedule |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4, cosine to 1e-6 |
| Schedule | Stage 2 200 epochs, Stage 3 50 epochs, batch 64 |
| Parameters | 4.2M full, 3.1M optimised |
| Edge latency | Jetson Nano 34 ms, Snapdragon 888 19 ms |
| Energy | 28 mJ per inference (Jetson Nano) |

## Locations

| Dataset | Classes | Access |
| --- | --- | --- |
| Kermany (UCSD) | CNV, DME, Drusen, Normal | https://data.mendeley.com/datasets/rscbjbr9sj/3 |
| Duke | AMD, DME, Normal | https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm |

Drop a manifest JSON (`{"items": [{"path": "frame.npy", "label": 0}, ...]}`) and
set `data.manifest` to use the real data; `scripts/prepare_data.sh` prints the
recipe.

## Ratings and advisories

- Research use only. The model supports a clinician; it does not replace one.
- The default data path is a deterministic synthetic OCT generator with
  recoverable per-class lesion signatures, so the repository runs offline
  without the credentialed corpora. Manuscript figures need the real data.
- Stage-1 backbone pretraining on external corpora is a hook, not an offline
  run; the acts that define the method are Stage 2 and Stage 3.
- The clinical alignment score reflects a human reader study; the test suite
  exercises the aggregation logic with a synthetic rater panel only.
- Engineering notes and per-equation provenance live under `docs/`.
