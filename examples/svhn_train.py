import json
import numpy as np
import sys
import torch
sys.path.append('../src') 
from torch.utils.data import TensorDataset, DataLoader


from model import D2NE as pb
from model.architectures import CNNClassifier_SVHN
from model import D2NE_trainer as tr

from helpers.config_loader import load_config

config = load_config()
exp_cfg = config["experiment"]
train_cfg = config["training"]
model_cfg = config["model"]
DIGITS = range(10) 


from torchvision.datasets import SVHN
from torchvision import transforms

transform = transforms.ToTensor()
train_set = SVHN(root='./data', split='train', download=True, transform=transform)
test_set = SVHN(root='./data', split='test', download=True, transform=transform)

train_X = torch.from_numpy(train_set.data).float()
train_y = torch.from_numpy(train_set.labels).long()
test_X = torch.from_numpy(test_set.data).float()
test_y = torch.from_numpy(test_set.labels).long()

# Filter for DIGITS
if DIGITS != list(range(10)):
        train_mask = np.isin(train_y, DIGITS)
        test_mask = np.isin(test_y, DIGITS)

        train_X, train_y = train_X[train_mask], train_y[train_mask]
        test_X, test_y = test_X[test_mask], test_y[test_mask]

# Normalize to [0, 1] range
train_X /= 255.0
test_X /= 255.0

# SVHN images are in format (N, C, H, W) = (N, 3, 32, 32)
# Keep 2D structure for CNN (don't flatten)
train_X = train_X.float()
test_X = test_X.float()


train_ds = TensorDataset(train_X, train_y)
test_ds = TensorDataset(test_X, test_y)
train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False)


model = pb.D2NE_model(config, CNNClassifier_SVHN)
trainer = tr.D2NETrainer(config, model, train_loader, test_loader)
accuracies = trainer.train()

result = {
    "exp_id": exp_cfg["exp_id"],
    "config": config,   # save FULL config
    "metrics": {
        "accuracies": accuracies
    }
}

with open(exp_cfg["out_file"], "a") as f:
    f.write(json.dumps(result) + "\n")

