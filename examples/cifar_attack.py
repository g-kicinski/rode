from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import foolbox as fb
import numpy as np
import torch
import yaml
from torchvision.datasets import CIFAR10


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from model import D2NE as pb
from model.architectures import SmallCifar10CNN


PGD_RESTART_SEEDS = (1729, 3253, 4861, 6421, 8081)
ATTACK_SPECS = (
    {
        "key": "fgsm_linf",
        "label": "FGSM (Linf)",
        "epsilons": (0.0025, 0.005),
        "factory": lambda: fb.attacks.LinfFastGradientAttack(random_start=False),
        "restarts": 1,
        "steps": 1,
    },
    {
        "key": "pgd_linf",
        "label": "PGD (Linf, 5 restarts)",
        "epsilons": (0.0025, 0.005),
        "factory": lambda: fb.attacks.LinfPGD(steps=40, random_start=True),
        "restarts": 5,
        "steps": 40,
    },
    {
        "key": "deepfool_linf",
        "label": "DeepFool (Linf)",
        "epsilons": (0.0025, 0.005),
        "factory": lambda: fb.attacks.LinfDeepFoolAttack(
            steps=50, candidates=10, overshoot=0.02, loss="logits"
        ),
        "restarts": 1,
        "steps": 50,
    },
    {
        "key": "pgd_l2",
        "label": "PGD (L2, 5 restarts)",
        "epsilons": (0.125, 0.25),
        "factory": lambda: fb.attacks.L2PGD(steps=50, random_start=True),
        "restarts": 5,
        "steps": 50,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pgd-restarts", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def clean_correct_mask(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    correct = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            logits = model(images[start:end].to(device))
            correct.append(logits.argmax(dim=1).cpu().eq(labels[start:end]))
    return torch.cat(correct)


def robust_percent(success: torch.Tensor) -> list[float]:
    return [
        100.0 * value.item()
        for value in (1.0 - success.float().mean(dim=1))
    ]


def run_attack_batched(
    spec: dict,
    fmodel: fb.PyTorchModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_correct: torch.Tensor,
    batch_size: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[list[float], list[list[float]]]:
    epsilons = spec["epsilons"]
    successes = (~clean_correct).unsqueeze(0).expand(len(epsilons), -1).clone()
    history = []
    if spec["key"].startswith("pgd_"):
        restart_seeds = PGD_RESTART_SEEDS[: spec["restarts"]]
    else:
        restart_seeds = (0,)

    for restart_index, restart_seed in enumerate(restart_seeds):
        set_seed(restart_seed, deterministic)
        attack = spec["factory"]()

        # Once every epsilon has already succeeded for an image, later restarts
        # cannot alter its robust status. Excluding it preserves the exact union
        # of successes while reducing repeated work.
        pending = (~successes).any(dim=0).nonzero(as_tuple=False).squeeze(1)
        for start in range(0, len(pending), batch_size):
            batch_indices = pending[start : start + batch_size]
            x = images[batch_indices].to(device)
            y = labels[batch_indices].to(device)
            _, _, success = attack(fmodel, x, y, epsilons=list(epsilons))
            successes[:, batch_indices] |= success.detach().cpu().bool()
            torch.cuda.empty_cache()

        values = robust_percent(successes)
        history.append(values)
        print(
            f"  restart {restart_index + 1}/{len(restart_seeds)} "
            + ", ".join(f"{value:.4f}%" for value in values),
            flush=True,
        )
        del attack
        torch.cuda.empty_cache()

    return robust_percent(successes), history


def attack_is_compatible(existing: dict | None, spec: dict) -> bool:
    if not existing or not existing.get("complete"):
        return False
    return (
        [float(value) for value in existing.get("epsilons", [])]
        == [float(value) for value in spec["epsilons"]]
        and int(existing.get("restarts", 1)) == int(spec["restarts"])
        and int(existing.get("steps", spec["steps"])) == int(spec["steps"])
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.pgd_restarts <= len(PGD_RESTART_SEEDS):
        raise ValueError(
            f"pgd-restarts must lie in [1, {len(PGD_RESTART_SEEDS)}]"
        )
    config_path = args.config.with_suffix(".yaml") if not args.config.suffix else args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    training_seed = int(exp_cfg.get("seed", 0))
    deterministic = bool(exp_cfg.get("deterministic", False))
    set_seed(training_seed, deterministic)

    test_data = CIFAR10(root=RELEASE_ROOT / "data", train=False, download=False)
    images = torch.from_numpy(test_data.data).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(test_data.targets, dtype=torch.long)
    if args.limit is not None:
        images = images[: args.limit]
        labels = labels[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(train_cfg["checkpoint"] + "_best.pt")
    if not checkpoint.is_absolute():
        checkpoint = RELEASE_ROOT / checkpoint
    model = pb.D2NE_model(config, SmallCifar10CNN)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    clean_correct = clean_correct_mask(model, images, labels, args.batch_size, device)
    clean = 100.0 * clean_correct.float().mean().item()
    print(f"Experiment: {exp_cfg['exp_id']}", flush=True)
    print(f"Checkpoint: {checkpoint.resolve()}", flush=True)
    print(f"Dataset size: {len(images)}", flush=True)
    print(f"Clean accuracy: {clean:.4f}%", flush=True)

    output = args.output
    if not output.is_absolute():
        output = RELEASE_ROOT / output
    output = output.resolve()
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("dataset_size") != len(images):
            result = {}
    else:
        result = {}
    result.update(
        {
            "experiment": exp_cfg["exp_id"],
            "seed": training_seed,
            "config": str(config_path.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "dataset": "CIFAR-10 test",
            "dataset_size": len(images),
            "batch_size": args.batch_size,
            "clean_accuracy": clean,
            "pgd_restart_seeds": list(PGD_RESTART_SEEDS[: args.pgd_restarts]),
            "aggregation": "per-image union of attack successes across restarts",
            "attacks": result.get("attacks", {}),
        }
    )

    for template in ATTACK_SPECS:
        spec = dict(template)
        if spec["key"].startswith("pgd_"):
            spec["restarts"] = args.pgd_restarts
            norm = "Linf" if spec["key"] == "pgd_linf" else "L2"
            spec["label"] = f"PGD ({norm}, {args.pgd_restarts} restart"
            if args.pgd_restarts != 1:
                spec["label"] += "s"
            spec["label"] += ")"
        existing = result["attacks"].get(spec["key"])
        if attack_is_compatible(existing, spec):
            print(f"Skipping completed attack {spec['label']}", flush=True)
            continue

        print(
            f"Starting {spec['label']} at epsilons {list(spec['epsilons'])}",
            flush=True,
        )
        started = time.time()
        robust, history = run_attack_batched(
            spec,
            fmodel,
            images,
            labels,
            clean_correct,
            args.batch_size,
            device,
            deterministic,
        )
        seconds = time.time() - started
        result["attacks"][spec["key"]] = {
            "label": spec["label"],
            "epsilons": list(spec["epsilons"]),
            "robust_accuracy": robust,
            "restarts": spec["restarts"],
            "restart_seeds": (
                list(PGD_RESTART_SEEDS[: spec["restarts"]])
                if spec["key"].startswith("pgd_")
                else []
            ),
            "restart_robust_accuracy": history,
            "steps": spec["steps"],
            "seconds": seconds,
            "complete": True,
        }
        atomic_write_json(output, result)
        print(
            f"Completed {spec['label']}: "
            + ", ".join(f"{value:.4f}%" for value in robust)
            + f" in {seconds / 60.0:.2f} min",
            flush=True,
        )
        torch.cuda.empty_cache()

    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
