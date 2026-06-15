from __future__ import annotations

from pathlib import Path

from vinexplainnet.treatment.loader import read_treatment

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "configs" / "experiment"


def test_include_merges_model_and_data() -> None:
    treatment = read_treatment(EXP / "main.conf")
    assert treatment.model.channels == (24, 40, 112, 320)
    assert treatment.data.name == "kermany"
    assert abs(treatment.train.lr - 1e-4) < 1e-12
    assert abs(treatment.ass.sigma_s - 10.0) < 1e-9


def test_ablation_overrides_through_include() -> None:
    treatment = read_treatment(EXP / "ablation_no_hafm.conf")
    assert treatment.model.use_fusion is False
    assert treatment.model.use_global is False
    assert treatment.model.channels == (24, 40, 112, 320)


def test_supplementary_swaps_dataset() -> None:
    treatment = read_treatment(EXP / "supplementary_duke.conf")
    assert treatment.data.name == "duke"
    assert len(treatment.data.classes) == 3
    assert treatment.num_classes == 3
