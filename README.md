# Rough Neural Ordinary Differential Equations

This repository contains the code used for experiments in the paper submitted to ESANN 2027.

Gabriel Kiciński<sup>1</sup>, Piotr Bogusław Mucha<sup>1</sup>, Jacek Cyranka<sup>2</sup>

<sup>1</sup> University of Warsaw – Institute of Applied Mathematics and Mechanics, Warsaw, Poland  
<sup>2</sup> Jagiellonian University – Faculty of Mathematics and Computer Science, Kraków, Poland

The experiments are configured through `examples/config.yaml`. The most important options, crucial for setting the architecture are:

```yaml
experiment:
  exp_id: divTGLANet_1
  out_file: "results.jsonl"

training:
  batch_size: 128
  epochs: 60
  lr: 0.001
  weight_decay: 5e-4

model:
  field_epsilon: 0.01
  use_cnn_only: false
  odeint_step_size: 0.05
  use_ripping_field: true
  augmented: true
  augmented_by_zeros: false
  div_free_field: "simple"
```

- `exp_id` - identifier of the experiment. 
- `out_file` - file where evaluation results are saved, in JSON Lines format.

- `field_epsilon` - hyperparameter controlling the strength of the sgn_sigma vector field.
- `use_cnn_only` - if `true`, disables the ODE layer and runs the pure CNN baseline.
- `odeint_step_size` - step size used by the ODE solver.
- `use_ripping_field` - enables the RODE transform (F0 + F1) mechanism.
- `augmented` - enables latent space augmentation.
- `augmented_by_zeros` - if `true`, uses standard zero augmentation instead of target-guided augmentation.
- `div_free_field` - selects the vector field parametrization used inside the ODE layer. Available options:
  - `"none"` - empty field; no ODE-based transformation is applied.
  - `"node"` - unconstrained Neural ODE field without the divergence-free constraint.
  - `"big"` - divergence-free field based on the parametrization used in *Neural Conservation Laws*; not used in the paper.
  - `"simple"` - used in RODE(Div1); simple divergence-free field parametrized by one scalar potential function; used in the main experiments.
  - `"simple3"` - used in RODE(Div3); simple divergence-free field parametrized by three scalar potential functions.


## Running experiments

Go to the `examples` directory:

```bash
cd examples
```

Train the model `cifar_train`, `svhn_train` or `mnist_train`:

```bash
python3 cifar_train.py
```

After training, run the corresponding attack evaluation:

```bash
python3 -u cifar_attack.py
```

## Output

Experiment results are saved to the file specified in `config.yaml`:

```yaml
out_file: "results.jsonl"
```

The file stores the evaluation results in JSON Lines format.

## Model variants
The `.sh` scripts provided in the `examples` folder correspond to the different architectures. All models for a given dataset share the same feature extractor, pooling layer and classification head; they differ only in the transformation applied to the latent features.

- `run_cnn.sh` - baseline CNN (Pure CNN in the paper): feature extractor followed directly by the linear classification head, with no continuous-time block.
- `run_node.sh` - baseline Neural ODE: the feature vector is evolved by an unconstrained neural vector field (single hidden layer, tanh), with no dimension augmentation and no RODE transform.
- `run_anode.sh` - baseline Augmented Neural ODE (Pure ANODE in the paper; Dupont et al., 2019): same vector field as above, but evolved in an augmented state space.
- `run_tglanode_learnable_scale.sh` - RODE(NODE) in the paper: RODE transform ($F_0 + F_1$) followed by an unconstrained tanh vector field $F_2$ with a learnable scale.
- `run_divtglanet_relu_learnable_scale.sh` - RODE(DivK) with $K=1$ or $3$ depending on the `div_free_field` variable in the config file: RODE transform followed by a divergence-free field $F_2$ built from ReLU potentials. For $K=3$ the gains of the components are learned.
- `run_rms.sh` - additional variant not reported in the paper (RMS-constrained gains).

## Reproducibility

Each experiment is repeated over ten training seeds:

    1337 2027 4099 5003 6007 7001 8009 9001 10007 11003

The `for SEED in ...` loops in the `.sh` scripts correspond to the seeds used in our work, so running a script as provided reproduces the reported mean ± standard deviation. The same hyperparameters are used for all models (Adam, learning rate 1e-3, weight decay 5e-4, batch size 128, 60 epochs).