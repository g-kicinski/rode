import json
import numpy as np
import sys
sys.path.append('../src') 
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import D2NE as pb
from model import D2NE_trainer as tr
from model.architectures import CNNClassifier_MNIST

from helpers.config_loader import load_config

def release_path(value):
    path = Path(value)
    return path if path.is_absolute() else RELEASE_ROOT / path


config = load_config()

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


from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.ToTensor()
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
test_set = MNIST(root='./data', train=False, download=True, transform=transform)

train_X = train_set.data.float()
train_y = train_set.targets
test_X = test_set.data.float()
test_y = test_set.targets


train_X /= 255.0
test_X /= 255.0

train_X = train_X.view(train_X.size(0), -1).float()
test_X = test_X.view(test_X.size(0), -1).float()

# For quicker testing, we can limit the number of samples used for training and testing
# train_X = train_X[:10000]
# train_y = train_y[:10000]
# test_X =  test_X[:2000]
# test_y =  test_y[:2000]

train_ds = TensorDataset(train_X, train_y)
test_ds = TensorDataset(test_X, test_y)
train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False)

model = pb.D2NE_model(config, CNNClassifier_MNIST)
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
