# Deviations

Every entry links a paper section and gives a justification. The science is
implemented exactly; these are environment/scope choices.

## D1 — Synthetic data stands in for the restricted real corpora
- Paper: §IV-A (Kermany / Duke / OCTID).
- The three datasets are public but large and registration/licensing varies. The
  default data path is a deterministic synthetic layered-retina OCT generator
  (`casting/synthetic.py`) whose per-class lesion signatures are linearly
  recoverable, so classification and attention explanations are exercised end to
  end offline. The real path is fully wired through `casting/sources.py`
  (manifest adapters with the published class maps); point it at downloaded data
  to reproduce the manuscript numbers.

## D2 — Stage-1 backbone pretraining is a hook, not an offline run
- Paper: §III-D-1 (100 ImageNet epochs + 50 medical-imaging epochs).
- External-corpus pretraining (ImageNet-1K, CheXpert) is out of scope for an
  offline smoke run. `direction/stages.py` defines the Stage-1 act and accepts a
  pretrained-weights manifest; when none is given it initialises the backbone
  and proceeds to Stage 2 / Stage 3 (the stages that define the method).

## D3 — Clinician ratings use a deterministic synthetic panel for tests
- Paper: §V-C, Tables IV/V (n = 23 ophthalmologists, Likert 1–5).
- Human ratings are not code-reproducible. `screening/clinical.py` computes the
  reported summary statistics (mean, bootstrap 95% CI) from a ratings file; the
  test suite feeds a deterministic synthetic 23-rater panel so the aggregation
  and CI logic are validated without fabricating clinical opinion.

## D4 — Differentiable bilateral filter for the ASS layer
- Paper: §III-A, Eq. 3 (`BilateralFilter(x, σ_s, σ_r)`).
- A classical bilateral filter is non-differentiable in its range kernel in a way
  that is awkward for autograd. `lens/speckle.py` uses a separable Gaussian
  spatial kernel with a range-weighted blend that preserves the Eq. 3 semantics
  (α=0 identity, α=1 maximal smoothing, edge-aware range term σ_r) and trains the
  learnable α end to end. Numerically faithful to the stated behaviour.
