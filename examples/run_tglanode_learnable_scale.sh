#!/bin/bash
#SBATCH --job-name=NODE_mnist
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mnist_%j.out
#SBATCH --error=logs/mnist_%j.err

echo "Running on $(hostname)"


source ~/venv/bin/activate

echo "Running on $(hostname)"

CONFIG="configs/tglanode_learnable_scale.yaml"

for SEED in 1337 2027 4099 5003 6007 7001 8009 9001 10007 11003
do

    # Modify YAML
    EXP_ID="tglanode_learnable_scale_${SEED}"
    sed -i \
        -e "s/^  exp_id:.*/  exp_id: ${EXP_ID}/" \
        -e "s/^  seed:.*/  seed: ${SEED}/" \
        "$CONFIG"

    echo "Running experiment with seed ${SEED}"

    python3 mnist_train.py --config "$CONFIG"
    python3 mnist_attack.py --config "$CONFIG"
    echo "Finished seed ${SEED}"
    echo "-----------------------------"
done