from __future__ import annotations

import json
from pathlib import Path

import torch

from vinexplainnet.casting.reels import make_loaders
from vinexplainnet.crew.rig import pick_device
from vinexplainnet.crew.vault import load_take
from vinexplainnet.cutting.quantize import MixedPrecisionQuantizer
from vinexplainnet.direction.runner import train_run
from vinexplainnet.feature import LogitsFeature, VINExplainNet, build_feature
from vinexplainnet.screening.booth import evaluate_reel
from vinexplainnet.screening.faithfulness import deletion_faithfulness, gini_sparsity
from vinexplainnet.screening.footprint import profile_footprint
from vinexplainnet.treatment.loader import read_treatment
from vinexplainnet.treatment.schema import Treatment

DEFAULT_TREATMENT = "configs/experiment/main.conf"


def _load_or_build(path: str) -> tuple[VINExplainNet, Treatment]:
    treatment = read_treatment(path)
    model = build_feature(treatment)
    take = Path(treatment.take_dir) / f"{treatment.run_name}.pt"
    if take.exists():
        state = load_take(take)
        model.load_state_dict(state["model"])
    return model, treatment


def train(treatment: str = DEFAULT_TREATMENT) -> str:
    spec = read_treatment(treatment)
    _, history, take_path = train_run(spec)
    last = history[-1] if history else {}
    return json.dumps({"take": str(take_path), "steps": len(history), "last": last})


def evaluate(treatment: str = DEFAULT_TREATMENT) -> str:
    model, spec = _load_or_build(treatment)
    device = pick_device(spec.train.device)
    _, val = make_loaders(spec)
    report = evaluate_reel(model, val, device)
    return json.dumps(report)


def explain(treatment: str = DEFAULT_TREATMENT) -> str:
    model, spec = _load_or_build(treatment)
    device = pick_device(spec.train.device)
    model.to(device)
    model.eval()
    _, val = make_loaders(spec)
    batch = next(iter(val))
    image = batch.images[:1].to(device)
    with torch.no_grad():
        frame = model(image)
    target = int(frame.logits.argmax(dim=1).item())
    explanation = frame.explanations[0, target]

    def prob(masked: torch.Tensor) -> float:
        with torch.no_grad():
            return float(model(masked).probs[0, target].item())

    faith = deletion_faithfulness(prob, image, explanation)
    gini = gini_sparsity(explanation)
    return json.dumps({"class": target, "faithfulness": faith, "gini": gini})


def optimize(treatment: str = DEFAULT_TREATMENT) -> str:
    model, spec = _load_or_build(treatment)
    sample = torch.randn(1, 1, spec.data.image_size, spec.data.image_size)
    before = profile_footprint(model, sample, runs=5)
    quantizer = MixedPrecisionQuantizer(spec.edge.quant_bits)
    report = quantizer.quantize(model)
    return json.dumps({"footprint": before, "quantization": report})


def export(treatment: str = DEFAULT_TREATMENT) -> str:
    model, spec = _load_or_build(treatment)
    model.eval()
    sample = torch.randn(1, 1, spec.data.image_size, spec.data.image_size)
    out = Path(spec.take_dir) / f"{spec.run_name}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        LogitsFeature(model),
        (sample,),
        str(out),
        input_names=["oct"],
        output_names=["logits"],
        dynamo=False,
    )
    return json.dumps({"onnx": str(out)})
