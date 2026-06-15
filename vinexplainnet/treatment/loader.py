from __future__ import annotations

from pathlib import Path
from typing import Any

from pyhocon import ConfigFactory

from vinexplainnet.treatment.schema import (
    AssSpec,
    DataSpec,
    EdgeSpec,
    ModelSpec,
    TrainSpec,
    Treatment,
)


def _pick(tree: Any, key: str, default: Any) -> Any:
    try:
        value = tree.get(key)
    except Exception:
        return default
    return default if value is None else value


def treatment_from_tree(conf: Any) -> Treatment:
    base = Treatment()
    channels = list(_pick(conf, "model.channels", list(base.model.channels)))
    chan4 = (int(channels[0]), int(channels[1]), int(channels[2]), int(channels[3]))
    classes = tuple(str(c) for c in _pick(conf, "data.classes", list(base.data.classes)))

    model = ModelSpec(
        channels=chan4,
        stem=int(_pick(conf, "model.stem", base.model.stem)),
        attention_dim=int(_pick(conf, "model.attention_dim", base.model.attention_dim)),
        hidden=int(_pick(conf, "model.hidden", base.model.hidden)),
        use_speckle=bool(_pick(conf, "model.use_speckle", base.model.use_speckle)),
        use_local=bool(_pick(conf, "model.use_local", base.model.use_local)),
        use_regional=bool(_pick(conf, "model.use_regional", base.model.use_regional)),
        use_global=bool(_pick(conf, "model.use_global", base.model.use_global)),
        use_fusion=bool(_pick(conf, "model.use_fusion", base.model.use_fusion)),
    )
    data = DataSpec(
        name=str(_pick(conf, "data.name", base.data.name)),
        classes=classes,
        image_size=int(_pick(conf, "data.image_size", base.data.image_size)),
        n_train=int(_pick(conf, "data.n_train", base.data.n_train)),
        n_val=int(_pick(conf, "data.n_val", base.data.n_val)),
        batch_size=int(_pick(conf, "data.batch_size", base.data.batch_size)),
        manifest=str(_pick(conf, "data.manifest", base.data.manifest)),
    )
    train = TrainSpec(
        lr=float(_pick(conf, "train.lr", base.train.lr)),
        min_lr=float(_pick(conf, "train.min_lr", base.train.min_lr)),
        weight_decay=float(_pick(conf, "train.weight_decay", base.train.weight_decay)),
        stage2_epochs=int(_pick(conf, "train.stage2_epochs", base.train.stage2_epochs)),
        stage3_epochs=int(_pick(conf, "train.stage3_epochs", base.train.stage3_epochs)),
        lambda_cls=float(_pick(conf, "train.lambda_cls", base.train.lambda_cls)),
        lambda_e=float(_pick(conf, "train.lambda_e", base.train.lambda_e)),
        lambda_r=float(_pick(conf, "train.lambda_r", base.train.lambda_r)),
        boundary_beta=float(_pick(conf, "train.boundary_beta", base.train.boundary_beta)),
        seed=int(_pick(conf, "train.seed", base.train.seed)),
        max_steps=int(_pick(conf, "train.max_steps", base.train.max_steps)),
        device=str(_pick(conf, "train.device", base.train.device)),
    )
    edge = EdgeSpec(
        prune_threshold_ratio=float(
            _pick(conf, "edge.prune_threshold_ratio", base.edge.prune_threshold_ratio)
        ),
        quant_bits=int(_pick(conf, "edge.quant_bits", base.edge.quant_bits)),
    )
    ass = AssSpec(
        sigma_s=float(_pick(conf, "ass.sigma_s", base.ass.sigma_s)),
        sigma_r=float(_pick(conf, "ass.sigma_r", base.ass.sigma_r)),
        kernel=int(_pick(conf, "ass.kernel", base.ass.kernel)),
    )
    return Treatment(
        run_name=str(_pick(conf, "run_name", base.run_name)),
        take_dir=str(_pick(conf, "take_dir", base.take_dir)),
        model=model,
        data=data,
        train=train,
        edge=edge,
        ass=ass,
    )


def read_treatment(path: str | Path) -> Treatment:
    conf = ConfigFactory.parse_file(str(path))
    return treatment_from_tree(conf)


def read_treatment_string(text: str) -> Treatment:
    conf = ConfigFactory.parse_string(text)
    return treatment_from_tree(conf)
