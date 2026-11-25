import torch
import foolbox as fb
from foolbox import accuracy, samples
import sys
sys.path.append('../src') 
import os


from model import D2NE as pb
from model.architectures import CNNClassifier_SVHN

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


# Shuffle test_X and test_y together
test_X = torch.from_numpy(test_set.data).float()
test_y = torch.from_numpy(test_set.labels).long()

images, labels = test_X.cuda().float() /255. , test_y.cuda().long()

subset_path = "subset_indices.pt"

if os.path.exists(subset_path):
        print(f"Loading subset indices from {subset_path} ...")
        indices = torch.load(subset_path)
else:
        print(f"Creating and saving subset indices to {subset_path} ...")
        seed = 42
        g = torch.Generator().manual_seed(seed)
        subset_size = 2000
        indices = torch.randperm(len(test_set), generator=g)[:subset_size]
        torch.save(indices, subset_path)

images_subsample = images[indices]
labels_subsample = labels[indices]

print("start evaluating attacks on SVHN ...")
model = pb.D2NE_model(config, CNNClassifier_SVHN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(train_cfg["checkpoint"]+"_best.pt", map_location=device))
# model.ripping_field.field_epsilon = 1e-3
model.eval()

fmodel = fb.PyTorchModel(model, bounds=(0, 1))

clean_acc = accuracy(fmodel, images, labels)
print(f"clean accuracy:  {clean_acc * 100:.1f} %")


# List of major attacks to evaluate
# Add CW attack to the list
attacks = [
        ("FGSM", fb.attacks.LinfFastGradientAttack()),
        ("PGD", fb.attacks.LinfPGD()),
        # ("L2PGD", fb.attacks.L2PGD()),
        ("DeepFool", fb.attacks.LinfDeepFoolAttack()),
        # ("BasicIterative", fb.attacks.LinfBasicIterativeAttack()),
        # ("Linfinity", fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack()),        
#        ("SaltAndPepper", fb.attacks.SaltAndPepperNoiseAttack()),
]
# Define different epsilons for each attack
attack_epsilons = {
        "FGSM": [0.005, 0.01],
        "PGD": [0.005, 0.01],
        # "L2PGD": [0.1, 0.2, 0.5],
        "DeepFool": [0.0025, 0.01],
        # "BasicIterative": [0.0025, 0.01, 0.02],
        # "Linfinity": [0.05, 0.1, 0.2],
#        "SaltAndPepper": [1.0, 3.0, 5.0],
}

batch_size = 250

def run_attack_batched(attack, fmodel, images, labels, batch_size=256, epsilons=None):
    all_success = []

    for i in range(0, len(images), batch_size):
        x = images[i:i+batch_size].cuda()
        y = labels[i:i+batch_size].cuda()

        _, _, success = attack(fmodel, x, y, epsilons=epsilons)
        all_success.append(success.detach().cpu())

        torch.cuda.empty_cache()

    return torch.cat(all_success, dim=1)

summary = []
for name, attack in attacks:
        epsilons = attack_epsilons.get(name, [0.1, 0.2, 0.3])
        print(f"\nEvaluating {name} with epsilons: {epsilons} ...")

        success = run_attack_batched(attack, fmodel, images, labels, batch_size=batch_size, epsilons=epsilons)
        robust_accuracy = 1 - success.float().mean(dim=1)
        summary.append((name, [acc.item() for acc in robust_accuracy]))
        print(f"Robust accuracy for {name}:")
        for eps, acc in zip(epsilons, robust_accuracy):
                print(f" norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")



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