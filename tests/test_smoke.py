from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vinexplainnet.studio import clapperboard

ROOT = Path(__file__).resolve().parents[1]
SMOKE = str(ROOT / "configs" / "experiment" / "_smoke.conf")


def test_smoke_train_then_evaluate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    trained = json.loads(clapperboard.train(SMOKE))
    assert trained["steps"] >= 1
    assert "total" in trained["last"]
    report = json.loads(clapperboard.evaluate(SMOKE))
    assert "accuracy" in report
    assert not math.isnan(report["accuracy"])


def test_smoke_explain_in_range() -> None:
    report = json.loads(clapperboard.explain(SMOKE))
    assert 0.0 <= report["faithfulness"] <= 1.0
    assert 0.0 <= report["gini"] < 1.0


def test_smoke_optimize_reports_reduction() -> None:
    report = json.loads(clapperboard.optimize(SMOKE))
    assert report["quantization"]["ratio"] < 1.0


def test_smoke_export_onnx(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    try:
        record = json.loads(clapperboard.export(SMOKE))
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"onnx export unavailable: {exc}")
    assert Path(record["onnx"]).exists()
