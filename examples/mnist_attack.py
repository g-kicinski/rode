import torch
import foolbox as fb
from foolbox import accuracy, samples
import sys
import numpy as np
import random
from tqdm import tqdm
sys.path.append('../src') 

from model import D2NE as pb
from model.architectures import CNNClassifier_MNIST

from helpers.config_loader import load_config

def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


config = load_config()
exp_cfg = config["experiment"]
train_cfg = config["training"]
model_cfg = config["model"]
BATCH_SIZE = train_cfg["batch_size"]
DIGITS = range(10) 

from torchvision.datasets import MNIST
from torchvision import transforms

print("Loading MNIST dataset...")

transform = transforms.ToTensor()
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
test_set = MNIST(root='./data', train=False, download=True, transform=transform)

print(f"Training set size: {len(train_set)}")
# Shuffle train_X and train_y together

train_X = train_set.data
train_y = train_set.targets

perm = torch.randperm(len(test_set))
test_X = test_set.data[perm]
test_y = test_set.targets[perm]


model = pb.D2NE_model(config, CNNClassifier_MNIST)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(train_cfg["checkpoint"]+"_best_" + str(exp_cfg['seed']) +'.pt', map_location=device))
# model.ripping_field.field_epsilon = 1e-3
model.eval()

fmodel = fb.PyTorchModel(model, bounds=(0, 1))

samples = 10000

images, labels = test_X[:samples].float().cuda() /255. , test_y[:samples].cuda()
labels = labels.long()

images_subsample, labels_subsample = images[:2000], labels[:2000]

print("Evaluating clean accuracy...")

clean_acc = accuracy(fmodel, images, labels)
print(f"clean accuracy:  {clean_acc * 100:.1f} %")

print("Evaluating adversarial attacks...")
# List of major attacks to evaluate
# Add CW attack to the list
PGD_RESTART_SEEDS = (1729, 3253, 4861, 6421, 8081)
attacks = [
        ("FGSM", fb.attacks.LinfFastGradientAttack(random_start=False)),
        ("PGD", fb.attacks.LinfPGD(steps=40, random_start=True)),
        ("L2PGD", fb.attacks.L2PGD(steps=50, random_start=True)),
        ("DeepFool", fb.attacks.LinfDeepFoolAttack(
            steps=50, candidates=10, overshoot=0.02, loss="logits"
        )),
        # ("Linfinity", fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack()),
        # ("SaltAndPepper", fb.attacks.SaltAndPepperNoiseAttack()),
]
# Define different epsilons for each attack
attack_epsilons = {
        "FGSM": [0.05, 0.1],
        "PGD": [0.05, 0.1],
        "L2PGD": [0.5, 1.0],
        "DeepFool": [0.05, 0.1],
        # "Linfinity": [0.2, 0.5, 0.75],
        # "SaltAndPepper": [1.0, 3.0, 5.0],
}



summary = []
for name, attack in attacks:
    epsilons = attack_epsilons[name]
    print(f"\nEvaluating {name} with epsilons: {epsilons} ...")

    # success shape = (len(epsilons), N)
    success_all = torch.zeros(
        (len(epsilons), len(images)),
        dtype=torch.bool,
    )

    restart_seeds = PGD_RESTART_SEEDS if name in ("PGD", "L2PGD") else (None,)

    for restart_seed in tqdm(restart_seeds, desc=f"Restart seeds for {name}"):

        if restart_seed is not None:
            torch.manual_seed(restart_seed)
            torch.cuda.manual_seed_all(restart_seed)
            np.random.seed(restart_seed)
            random.seed(restart_seed)

            # create a fresh attack so random initialization changes
            if name == "PGD":
                attack = fb.attacks.LinfPGD(steps=40, random_start=True)
            else:
                attack = fb.attacks.L2PGD(steps=50, random_start=True)

        for start in tqdm(range(0, len(images), BATCH_SIZE), desc="Processing batches"):
            end = min(start + BATCH_SIZE, len(images))

            x = images[start:end]
            y = labels[start:end]

            _, _, success = attack(
                fmodel,
                x,
                y,
                epsilons=epsilons,
            )

            success_all[:, start:end] |= success.cpu()

    robust_accuracy = 1 - success_all.float().mean(dim=1)

    summary.append((name, [a.item() for a in robust_accuracy]))

    print(f"Robust accuracy for {name}:")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f" norm ≤ {eps:<6}: {100*acc.item():4.1f}%")


print("\n=== Summary of Robust Accuracies (CSV format) ===")
# Prepare header
all_epsilons = []
for name, _ in summary:
        epsilons = attack_epsilons.get(name, [0.1, 0.2, 0.3])
        all_epsilons.append(epsilons)

# Find max number of epsilons for header
max_eps = max(len(eps) for eps in all_epsilons)
header = ["Attack"] + [f"eps_{i+1}" for i in range(max_eps)]
print(",".join(header))

# Print clean accuracy as a row
print(f"Clean,{clean_acc*100:.2f}" + ","*(max_eps-1))

# Print each attack's results as a row
for idx, (name, accs) in enumerate(summary):
        epsilons = all_epsilons[idx]
        row = [name] + [f"{a*100:.2f}" for a in accs]
        # Pad with empty cells if fewer epsilons
        row += ["" for _ in range(max_eps - len(accs))]
        print(",".join(row))

# Save results to CSV file
import csv
import os
csv_filename = f"{exp_cfg["out_file"]} attack_results.csv"
csv_path = os.path.join(os.path.dirname(__file__), csv_filename)

write_header = not os.path.exists(csv_path)
with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
                writer.writerow(header)
        writer.writerow(["Clean", f"{config['experiment']['exp_id']}"])
        # Write clean accuracy row
        writer.writerow(["Clean", f"{clean_acc*100:.2f}"] + ["" for _ in range(max_eps-1)])
        # Write each attack's results as a row
        for idx, (name, accs) in enumerate(summary):
                row = [name] + [f"{a*100:.2f}" for a in accs]
                row += ["" for _ in range(max_eps - len(accs))]
                print(row)
                writer.writerow(row)

print(f"Results saved to {csv_path}")