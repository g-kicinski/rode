from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
import torchode as tode
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm


class _FixedWeightScale(nn.Module):
    """Scale a spectrally normalized weight without scaling its bias."""

    def __init__(self, coefficient):
        super().__init__()
        if coefficient <= 0:
            raise ValueError("spectral coefficient must be positive")
        self.register_buffer("coefficient", torch.tensor(float(coefficient)))

    def forward(self, weight):
        return self.coefficient * weight


def bounded_spectral_norm(layer, coefficient, n_power_iterations=1):
    spectral_norm(
        layer,
        name="weight",
        n_power_iterations=int(n_power_iterations),
    )
    parametrize.register_parametrization(
        layer,
        "weight",
        _FixedWeightScale(coefficient),
    )
    return layer


def normalize_cnn_convolutions(module, coefficients, n_power_iterations=1):
    convolutions = [layer for layer in module.modules() if isinstance(layer, nn.Conv2d)]
    if len(convolutions) != len(coefficients):
        raise ValueError(
            f"Expected {len(convolutions)} CNN spectral bounds, got {len(coefficients)}"
        )
    for layer, coefficient in zip(convolutions, coefficients):
        bounded_spectral_norm(layer, coefficient, n_power_iterations)

def labels_to_target_points(labels):
    """
    Convert arbitrary labels to {-1, +1} binary vectors.
    """
    encoder  = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)
    n_new = int(np.ceil(np.log2(num_classes)))

    bits = ((labels_encoded[:, None] >> np.arange(n_new)) & 1)

    target_points = bits * 2 - 1

    return target_points, encoder


class ODEFunc(nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field

    def forward(self, t, y, args=None):
        if getattr(self.field, "uses_time", False):
            return self.field(t, y)
        return self.field(y)


class ODEFunc2(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

        # Optional: better initialization (used in the paper style)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t, x):
        return self.net(x)

class RippingField(nn.Module):
    def __init__(self, num_features, field_epsilon):
        super().__init__()
        self.num_features = num_features
        self.field_epsilon = field_epsilon
    
    def f(self, y, epsilon=None):
        if epsilon is None:
            epsilon = self.field_epsilon
        return y / (y ** 2 + epsilon) ** 0.5

    def forward(self, x, epsilon=None):
        return self.f(x, epsilon)

from .div_free_fields import *

DIV_FREE_FIELD_MAP = {
    "none": EmptyField,
    "node": SimpleNet, # a node-based field without the div-free constraint
    "big": DivFreeField, # the div-free field used in the paper "Neural Conservation Laws"
    "simple": DivFreeFieldSimple, # a simpler div-free field with only one scalar function parametrizing it
    "simple_multi": DivFreeFieldMultiPotential,
    "simple_sequential": DivFreeFieldSequential,
    "simple3": DivFreeFieldSimple5, # a simpler div-free field with five scalar functions parametrizing it
}

def build_div_free_field(cfg, dim):
    cls = DIV_FREE_FIELD_MAP[cfg["div_free_field"]]
    if cls == DivFreeFieldSimple5:
        return cls(
            dim, 
            hidden_dim = cfg["div_free_field_hidden_size"], 
            num_non_zero=cfg["div_free5_num_non_zero"]
            )
    if cls == DivFreeFieldSimple:
        return cls(
            dim,
            hidden_dim=cfg.get("div_free_simple_hidden_size", 64),
            activation=cfg.get("div_free_simple_activation", "relu"),
            leaky_relu_slope=cfg.get("div_free_simple_leaky_relu_slope", 0.1),
            softplus_beta=cfg.get("div_free_simple_softplus_beta", 5.0),
            num_hidden_layers=cfg.get("div_free_simple_num_hidden_layers", 1),
            learnable_mixing=cfg.get("div_free_simple_learnable_mixing", False),
            normalize_mixing=cfg.get("div_free_simple_normalize_mixing", False),
            learnable_scale=cfg.get("div_free_simple_learnable_scale", False),
            initial_scale=cfg.get("div_free_simple_initial_scale", 1e-2),
            bounded_scale_max=cfg.get("div_free_simple_bounded_scale_max"),
            normalized_mixing_parameterization=cfg.get(
                "div_free_simple_normalized_mixing_parameterization", "cayley"
            ),
        )
    if cls == DivFreeFieldMultiPotential:
        return cls(
            dim,
            hidden_dim=cfg.get("div_free_simple_hidden_size", 64),
            activation=cfg.get("div_free_simple_activation", "leaky_relu"),
            leaky_relu_slope=cfg.get("div_free_simple_leaky_relu_slope", 0.1),
            softplus_beta=cfg.get("div_free_simple_softplus_beta", 5.0),
            num_hidden_layers=cfg.get("div_free_simple_num_hidden_layers", 1),
            num_potentials=cfg.get("div_free_multi_num_potentials", 3),
            normalize_mixing=cfg.get("div_free_simple_normalize_mixing", True),
            initial_scale=cfg.get("div_free_simple_initial_scale", 1.0),
            learnable_mixing=cfg.get("div_free_multi_learnable_mixing", False),
            normalized_mixing_parameterization=cfg.get(
                "div_free_simple_normalized_mixing_parameterization", "cayley"
            ),
            learnable_gains=cfg.get("div_free_multi_learnable_gains", False),
            total_gain=cfg.get("div_free_multi_total_gain", 13.0),
        )
    if cls == DivFreeFieldSequential:
        return cls(
            dim,
            hidden_dim=cfg.get("div_free_simple_hidden_size", 64),
            activation=cfg.get("div_free_simple_activation", "leaky_relu"),
            leaky_relu_slope=cfg.get("div_free_simple_leaky_relu_slope", 0.1),
            softplus_beta=cfg.get("div_free_simple_softplus_beta", 5.0),
            num_hidden_layers=cfg.get("div_free_simple_num_hidden_layers", 1),
            num_stages=cfg.get("div_free_sequential_num_stages", 2),
            normalize_mixing=cfg.get("div_free_simple_normalize_mixing", False),
            initial_scale=cfg.get("div_free_simple_initial_scale", 0.6),
        )
    if cls == SimpleNet:
        return cls(
            dim,
            hidden_dim=cfg.get("node_hidden_size", 64),
            activation=cfg.get("node_activation", "tanh"),
            leaky_relu_slope=cfg.get("node_leaky_relu_slope", 0.1),
            softplus_beta=cfg.get("node_softplus_beta", 5.0),
            learnable_scale=cfg.get("node_learnable_scale", False),
            initial_scale=cfg.get("node_initial_scale", 1.0),
        )
    return cls(dim)

class D2NE_model(nn.Module):
    def __init__(self, cfg, classifier_cls):
        super().__init__()

        self.cfg = cfg["model"]

        self.num_features = self.cfg["num_features"]
        self.use_cnn_only = self.cfg["use_cnn_only"]
        self.use_ripping_field = self.cfg["use_ripping_field"]
        self.use_linear_1 = self.cfg["use_linear_1"]
        self.linear_1_size = self.cfg["linear_1_size"]
        self.augmented = self.cfg["augmented"]
        self.augmented_by_zeros = self.cfg["augmented_by_zeros"]
        self.target_softmax_temperature = float(
            self.cfg.get("target_softmax_temperature", 1.0)
        )
        if self.target_softmax_temperature <= 0:
            raise ValueError("target_softmax_temperature must be positive")

        target_classes, _ = labels_to_target_points(range(self.cfg["num_features"]))
        self.register_buffer(
            "target_classes",
            torch.as_tensor(target_classes, dtype=torch.float32)
        )
        self.num_target = target_classes.shape[1]

        self.n = self.num_target + self.num_features if not self.use_cnn_only else self.num_features

        self.classifier = classifier_cls(self.cfg["num_classes"])
        if self.cfg.get("cnn_spectral_normalization", False):
            normalize_cnn_convolutions(
                self.classifier,
                self.cfg.get("cnn_spectral_bounds", [1.2, 1.5, 2.0, 2.0, 2.0]),
                self.cfg.get("spectral_norm_power_iterations", 1),
            )
        if self.use_cnn_only and self.cfg.get(
            "classifier_spectral_normalization", False
        ):
            cnn_classifier = getattr(self.classifier, "classifier", None)
            if not isinstance(cnn_classifier, nn.Linear):
                raise ValueError(
                    "CNN-only classifier spectral normalization requires "
                    "classifier.classifier to be an nn.Linear layer"
                )
            bounded_spectral_norm(
                cnn_classifier,
                self.cfg.get("classifier_spectral_bound", 2.0),
                self.cfg.get("spectral_norm_power_iterations", 1),
            )
        if self.use_linear_1:
            self.layer1 = nn.Linear(self.num_features, self.linear_1_size)
        if self.augmented:
            dim = self.linear_1_size + self.num_target
        else:
            dim = self.linear_1_size
        if not self.use_cnn_only:
            self.layer2 = nn.Linear(dim, self.cfg["num_classes"])
            if self.cfg.get("classifier_spectral_normalization", False):
                bounded_spectral_norm(
                    self.layer2,
                    self.cfg.get("classifier_spectral_bound", 2.0),
                    self.cfg.get("spectral_norm_power_iterations", 1),
                )
        self.ripping_field = RippingField(self.num_target, self.cfg["field_epsilon"])
        
        self.div_free_field = build_div_free_field(self.cfg, dim)
        self.func = ODEFunc(self.div_free_field)

        self.ode_integrator = self.cfg.get("ode_integrator", "euler")
        self.semi_implicit_iterations = int(self.cfg.get("semi_implicit_iterations", 1))
        if self.semi_implicit_iterations < 1:
            raise ValueError("semi_implicit_iterations must be at least one")
        if self.ode_integrator not in {"euler", "semi_implicit"}:
            raise ValueError(f"Unsupported ode_integrator: {self.ode_integrator}")
        if self.ode_integrator == "semi_implicit":
            if not isinstance(self.div_free_field, DivFreeFieldSimple):
                raise ValueError("semi_implicit currently requires div_free_field=simple")
            if self.div_free_field.learnable_mixing:
                raise ValueError("semi_implicit currently requires a fixed mixing matrix")
            self.register_buffer(
                "semi_implicit_basis",
                self._canonical_basis_from_skew(self.div_free_field.mixing_matrix()),
            )

        term = tode.ODETerm(self.func)
        step_method = tode.Euler(term=term)
        controller = tode.FixedStepController()

        self.ivp_solver = tode.AutoDiffAdjoint(step_method, controller)

        self.field_epsilon = self.cfg["field_epsilon"]
        self.ode_step = self.cfg["odeint_step_size"]

        self.gif_steps = 2

    @staticmethod
    def _canonical_basis_from_skew(mixing):
        """Return an orthogonal basis grouped as the q and p coordinates of M."""
        import scipy.linalg

        if mixing.shape[0] % 2:
            raise ValueError("semi_implicit requires an even state dimension")
        _, basis = scipy.linalg.schur(mixing.detach().cpu().numpy(), output="real")
        q_indices = list(range(0, mixing.shape[0], 2))
        p_indices = list(range(1, mixing.shape[0], 2))
        basis = torch.from_numpy(basis[:, q_indices + p_indices])
        return basis.to(dtype=mixing.dtype, device=mixing.device)

    def _semi_implicit_solve(self, x_node):
        """Fixed-step partitioned semi-implicit Euler in canonical skew coordinates."""
        steps_float = 1.0 / float(self.ode_step)
        steps = int(round(steps_float))
        if not np.isclose(steps * float(self.ode_step), 1.0):
            raise ValueError("semi_implicit requires odeint_step_size to divide [0, 1]")

        basis = self.semi_implicit_basis
        state = x_node @ basis
        half = state.shape[1] // 2
        step_size = float(self.ode_step)

        for _ in range(steps):
            q_old, p_old = state[:, :half], state[:, half:]
            p_next = p_old
            for _ in range(self.semi_implicit_iterations):
                predictor = torch.cat([q_old, p_next], dim=1)
                predictor_field = self.div_free_field(predictor @ basis.T) @ basis
                p_next = p_old + step_size * predictor_field[:, half:]

            predictor = torch.cat([q_old, p_next], dim=1)
            predictor_field = self.div_free_field(predictor @ basis.T) @ basis
            q_next = q_old + step_size * predictor_field[:, :half]
            state = torch.cat([q_next, p_next], dim=1)

        return state @ basis.T


    def forward(self, x):
        if self.use_cnn_only:
            return self.classifier(x)[0]
        
        _, logits = self.classifier(x) 
        probs = torch.softmax(logits / self.target_softmax_temperature, dim=1)
        target_projection = torch.matmul(probs, self.target_classes)

        if self.use_ripping_field:
            target_projection = target_projection + self.ripping_field(target_projection)

        if self.use_linear_1:
            transformed = self.layer1(logits)
        else:
            transformed = logits
        
        if self.augmented and not self.augmented_by_zeros:
            x_node = torch.cat([transformed, target_projection], dim=1)
        elif self.augmented and self.augmented_by_zeros:
            zeros = torch.zeros_like(target_projection)
            x_node = torch.cat([transformed, zeros], dim=1)
        else:
            x_node = transformed
        
        t = torch.tensor([0.0, 1.0], device=x.device).expand(x_node.size(0), -1)

        if self.div_free_field.__class__ != EmptyField:
            if self.ode_integrator == "semi_implicit":
                scores = self._semi_implicit_solve(x_node)
            else:
                solution = self.ivp_solver.solve(
                    tode.InitialValueProblem(y0=x_node, t_eval=t),
                    dt0=torch.tensor([self.ode_step], device=x.device)
                )
                scores = solution.ys[:, -1, :]
        else:
            scores = x_node
        return self.layer2(scores)
