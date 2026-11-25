import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from model import D2NE as pb
from model import D2NE_trainer as tr
from model.architectures import SmallCifar10CNN


if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/cifar_train.py CONFIG.yaml")

config_path = Path(sys.argv[1])
if not config_path.suffix:
    config_path = config_path.with_suffix(".yaml")
config_path = config_path.resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))


def release_path(value):
    path = Path(value)
    return path if path.is_absolute() else RELEASE_ROOT / path

seed = config.get("experiment", {}).get("seed")
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if config.get("experiment", {}).get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

exp_cfg = config["experiment"]
train_cfg = config["training"]
model_cfg = config["model"]
out_file = release_path(exp_cfg["out_file"])
checkpoint = release_path(train_cfg["checkpoint"])
out_file.parent.mkdir(parents=True, exist_ok=True)
checkpoint.parent.mkdir(parents=True, exist_ok=True)
exp_cfg["out_file"] = str(out_file)
train_cfg["checkpoint"] = str(checkpoint)
print(exp_cfg)
print(train_cfg)
print(model_cfg)

train_data = CIFAR10(
    root=RELEASE_ROOT / "data",
    train=True,
    download=True,
)

test_data = CIFAR10(
    root=RELEASE_ROOT / "data",
    train=False,
    download=True,
)

# Convert images from:
# [N, 32, 32, 3], uint8
# to:
# [N, 3, 32, 32], float32 in [0, 1]
train_X = (
    torch.from_numpy(train_data.data)
    .permute(0, 3, 1, 2)
    .float()
    / 255.0
)

test_X = (
    torch.from_numpy(test_data.data)
    .permute(0, 3, 1, 2)
    .float()
    / 255.0
)

train_y = torch.tensor(train_data.targets, dtype=torch.long)
test_y = torch.tensor(test_data.targets, dtype=torch.long)

train_ds = TensorDataset(train_X, train_y)
test_ds = TensorDataset(test_X, test_y)

train_loader = DataLoader(
    train_ds,
    batch_size=train_cfg["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=train_cfg["batch_size"],
    shuffle=False
)

print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)


model = pb.D2NE_model(config, SmallCifar10CNN)
trainer = tr.D2NETrainer(config, model, train_loader, test_loader)

print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Device:", next(model.parameters()).device)
print("Batches:", len(train_loader))

accuracies = trainer.train()

result = {
    "exp_id": exp_cfg["exp_id"],
    "config": config,  
    "metrics": {
        "accuracies": accuracies
    }
}

with out_file.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(result) + "\n")
