from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssSpec:
    sigma_s: float = 10.0
    sigma_r: float = 0.1
    kernel: int = 5


@dataclass(frozen=True)
class ModelSpec:
    channels: tuple[int, int, int, int] = (24, 40, 112, 320)
    stem: int = 16
    attention_dim: int = 80
    hidden: int = 128
    use_speckle: bool = True
    use_local: bool = True
    use_regional: bool = True
    use_global: bool = True
    use_fusion: bool = True


@dataclass(frozen=True)
class DataSpec:
    name: str = "kermany"
    classes: tuple[str, ...] = ("CNV", "DME", "Drusen", "Normal")
    image_size: int = 224
    n_train: int = 256
    n_val: int = 64
    batch_size: int = 64
    manifest: str = ""


@dataclass(frozen=True)
class TrainSpec:
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    stage2_epochs: int = 200
    stage3_epochs: int = 50
    lambda_cls: float = 1.0
    lambda_e: float = 0.1
    lambda_r: float = 5e-4
    boundary_beta: float = 1.0
    seed: int = 1234
    max_steps: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class EdgeSpec:
    prune_threshold_ratio: float = 0.3
    quant_bits: int = 8


@dataclass(frozen=True)
class Treatment:
    run_name: str = "vinexplainnet_main"
    take_dir: str = "takes"
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    edge: EdgeSpec = field(default_factory=EdgeSpec)
    ass: AssSpec = field(default_factory=AssSpec)

    @property
    def num_classes(self) -> int:
        return len(self.data.classes)
