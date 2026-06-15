# Repo Plan

A motion-picture-studio metaphor: an OCT B-scan enters as raw footage and the
pipeline turns it into a finished, captioned "feature" (a prediction plus its
visual explanation). Flat package, no `src/`.

## Tree

```
vinexplainnet/
  treatment/        config schema + HOCON loader (the film "treatment")
    schema.py         frozen dataclasses (Treatment, ModelSpec, DataSpec, ...)
    loader.py         pyhocon read + include resolution -> Treatment
  casting/          data: who appears on screen
    synthetic.py      SyntheticOCT layered-retina generator
    sources.py        Kermany/Duke/OCTID manifest adapters, DATASET_SPECS
    degrade.py        ConsumerDegradation (blur/speckle/resolution/contrast)
    augment.py        ConsumerAugment (geometric/intensity/noise/OCT)
    reels.py          Reel dataset + make_loaders
  lens/             backbone optics
    speckle.py        AdaptiveSpeckleSuppression (Eq.3)
    backbone.py       EfficientBackbone DWS-Conv pyramid (Eq.2)
  lighting/         where attention falls (HAFM)
    streams.py        Local/Regional/Global attention streams (Eq.4-9)
    fusion.py         HierarchicalAttentionFusion (Eq.10-11)
  framing/          composition of the shot
    heads.py          PredictionHead (Eq.12-13)
  captions/         the visual explanation overlay
    generator.py      ExplanationGenerator (Eq.14-16)
  notes/            director's notes = objectives
    classification.py weighted_bce (Eq.17)
    explanation.py    coherence/sparsity/boundary (Eq.18-21)
    regularization.py l2_penalty (Eq.22)
    objective.py      TotalObjective (Eq.1)
  cutting/          the cutting room = edge optimisation
    prune.py          channel_importance, structured_prune (Eq.24)
    quantize.py       MixedPrecisionQuantizer (Eq.25)
  direction/        running the shoot
    schedule.py       cosine_lr (Eq.23)
    clapper.py        optimizer + atomic checkpoint
    stages.py         three-stage plan
    runner.py         Director.shoot (Algorithm 1)
  screening/        the test screening + critics
    metrics.py        classification_report
    faithfulness.py   deletion_faithfulness, gini_sparsity (Eq.26)
    clinical.py       clinical_alignment
    significance.py   mcnemar_test, bonferroni, bootstrap_ci
    robustness.py     degradation_sweep
    transfer.py       cross_dataset
    footprint.py      profile_footprint
  crew/             below-the-line utilities
    seed.py, logbook.py, vault.py (io), rig.py (device), reel_types.py
  studio/           CLI
    clapperboard.py   argh verbs
    __main__.py       dispatch
  feature.py        VINExplainNet assembled model + build_feature
configs/
  model/*.conf  data/*.conf  experiment/{main,ablation_*,supplementary_*,_smoke}.conf
tests/            varied pytest suite (see implementation-map test matrix)
docs/             project-context / implementation-map / deviations / repo-plan
scripts/          launch_train.sh / launch_eval.sh / prepare_data.sh
```

## Module responsibilities
- `treatment` owns all typed config; nothing else parses HOCON.
- `casting` is the only place that produces tensors-from-data; it never imports model code.
- `lens`/`lighting`/`framing`/`captions` are pure `nn.Module`s; no IO, no config parsing.
- `notes` are pure tensor functions/modules; no IO.
- `direction` is the only writer of checkpoints; `screening` is read-only over predictions.
- `studio` is the only argh surface; `crew` is dependency-free helpers.

## Dependencies (pinned ranges)
- python >=3.10,<3.14
- torch >=2.2 (dev verified 2.9)
- numpy >=1.26, scipy >=1.11, scikit-learn >=1.3
- argh >=0.31 (CLI), pyhocon >=0.3.61 (HOCON config)
- dev: ruff, black, isort, mypy, pytest

## Test coverage matrix
shape | ASS α∈[0,1] + α=0 identity | HAFM fusion-weights sum-to-1 invariant |
coherence/sparsity/boundary closed-form | deletion-faithfulness monotonicity |
Gini bounds | metrics vs sklearn | McNemar/bootstrap vs scipy | pruning reduces
channels | int8 quant parity | overfit-single-batch | gradient-flow to backbone |
determinism | HOCON include merge | house-style (no-comment/no-docstring/
forbidden-phrase/no-emoji via ast+tokenize) | e2e 2-step training smoke.
