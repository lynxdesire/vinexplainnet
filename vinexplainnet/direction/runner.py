from __future__ import annotations

from pathlib import Path

import torch

from vinexplainnet.casting.augment import ConsumerAugment
from vinexplainnet.casting.reels import Reel, make_loaders
from vinexplainnet.crew.logbook import get_logger
from vinexplainnet.crew.reel_types import Batch
from vinexplainnet.crew.rig import pick_device
from vinexplainnet.crew.seed import set_seed
from vinexplainnet.crew.vault import save_take
from vinexplainnet.direction.clapper import make_optimizer, set_lr
from vinexplainnet.direction.schedule import cosine_lr
from vinexplainnet.feature import VINExplainNet, build_feature
from vinexplainnet.notes.classification import class_balance_weights
from vinexplainnet.notes.objective import TotalObjective
from vinexplainnet.treatment.schema import Treatment


class Director:
    def __init__(self, treatment: Treatment) -> None:
        self.treatment = treatment
        self.device = pick_device(treatment.train.device)
        self.logger = get_logger("vinexplainnet.direction")

    def shoot(
        self, model: VINExplainNet, reel: Reel, max_steps: int, augment_on: bool = True
    ) -> list[dict[str, float]]:
        model.to(self.device)
        model.train()
        spec = self.treatment.train
        pos_weight = class_balance_weights(reel.labels, self.treatment.num_classes).to(self.device)
        optimizer = make_optimizer(model, spec.lr, spec.weight_decay)
        augment = ConsumerAugment(spec.seed)
        objective = TotalObjective(spec)
        history: list[dict[str, float]] = []
        step = 0
        while step < max_steps:
            for batch in reel:
                if step >= max_steps:
                    break
                if augment_on:
                    images = augment(batch.images).to(self.device)
                else:
                    images = batch.images.to(self.device)
                labels = batch.labels.to(self.device)
                frame = model(images)
                loss, report = objective.evaluate(
                    frame, Batch(images, labels), model.parameters(), pos_weight
                )
                lr = cosine_lr(step, max_steps, spec.lr, spec.min_lr)
                set_lr(optimizer, lr)
                optimizer.zero_grad()
                torch.autograd.backward(loss)
                optimizer.step()
                report["lr"] = lr
                report["step"] = float(step)
                history.append(report)
                step += 1
        return history


def resolve_steps(treatment: Treatment, reel: Reel) -> int:
    if treatment.train.max_steps > 0:
        return treatment.train.max_steps
    return len(reel)


def train_run(treatment: Treatment) -> tuple[VINExplainNet, list[dict[str, float]], Path]:
    set_seed(treatment.train.seed)
    train_reel, _ = make_loaders(treatment)
    model = build_feature(treatment)
    director = Director(treatment)
    steps = resolve_steps(treatment, train_reel)
    history = director.shoot(model, train_reel, steps)
    take_path = Path(treatment.take_dir) / f"{treatment.run_name}.pt"
    save_take(
        take_path,
        {"model": model.state_dict(), "seed": treatment.train.seed, "steps": steps},
    )
    return model, history, take_path
