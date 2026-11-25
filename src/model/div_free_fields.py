from torch.func import functional_call, jacrev, vmap
from torch import nn
import torch
import torch.nn.functional as F

class EmptyField(nn.Module):
    def __init__(self, n):
        super().__init__()
    def forward(self, x):
        return torch.zeros_like(x)
    
class SimpleNet(nn.Module):
    def __init__(
        self,
        n,
        hidden_dim=64,
        activation="tanh",
        leaky_relu_slope=0.1,
        softplus_beta=5.0,
        learnable_scale=False,
        initial_scale=1.0,
    ):
        super().__init__()
        self.n = n
        self.learnable_scale = learnable_scale

        if activation == "tanh":
            activation_layer = nn.Tanh()
        elif activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "leaky_relu":
            activation_layer = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        elif activation == "softplus":
            activation_layer = nn.Softplus(beta=softplus_beta)
        else:
            raise ValueError(f"Unsupported SimpleNet activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(n, hidden_dim),
            activation_layer,
            nn.Linear(hidden_dim, n),
        )

        if self.learnable_scale:
            initial_scale_tensor = torch.tensor(float(initial_scale))
            self.scale_raw = nn.Parameter(torch.log(torch.expm1(initial_scale_tensor)))
        else:
            self.scale = float(initial_scale)

    def current_scale(self):
        if self.learnable_scale:
            return F.softplus(self.scale_raw)
        return self.scale

    def forward(self, x):
        return self.current_scale() * self.net(x)
    
class RippingActivation(nn.Module):
    def __init__(self, field_epsilon=0.01):
        super().__init__()
        self.field_epsilon = field_epsilon
    
    def forward(self, x):
        return x / (x**2 + self.field_epsilon).sqrt()

class DivFreeField(nn.Module):
    def __init__(self, n, hidden_dim = 16):
        super().__init__()
        self.n = n
        self.training = True
        self.scale = 1e-2
        self.net = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def A(self, params, x):
        def f(params, x):
            return functional_call(self.net, params, (x,)) # * self.scale
        
        J = jacrev(f, argnums=1)(params, x)
        return J - J.T

    def u_single(self, params, x):
        def A_flat(params, x):
            return self.A(params, x).reshape(-1)

        D = x.numel()
        dA = jacrev(A_flat, argnums=1)(params, x)
        dA = dA.reshape(D, D, D)

        return vmap(torch.trace)(dA)


    def forward(self, x):
        params = dict(self.net.named_parameters())
        if not self.training:
            with torch.no_grad():   # <-- use this
                return vmap(lambda xi: self.u_single(params, xi))(x)

        return vmap(lambda xi: self.u_single(params, xi))(x)

class DivFreeFieldSimple(nn.Module):
    def __init__(
        self,
        n,
        hidden_dim=64,
        activation="relu",
        leaky_relu_slope=0.1,
        softplus_beta=5.0,
        num_hidden_layers=1,
        learnable_mixing=False,
        normalize_mixing=False,
        learnable_scale=False,
        initial_scale=1e-2,
        bounded_scale_max=None,
        normalized_mixing_parameterization="cayley",
    ):
        super().__init__()
        self.n = n
        self.learnable_mixing = learnable_mixing
        self.normalize_mixing = normalize_mixing
        self.learnable_scale = learnable_scale
        self.bounded_scale_max = bounded_scale_max
        self.normalized_mixing_parameterization = normalized_mixing_parameterization

        if self.normalized_mixing_parameterization not in {"cayley", "normalized_skew"}:
            raise ValueError(
                "normalized_mixing_parameterization must be cayley or normalized_skew"
            )

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        def make_activation():
            if activation == "relu":
                return nn.ReLU()
            if activation == "leaky_relu":
                return nn.LeakyReLU(negative_slope=leaky_relu_slope)
            if activation == "softplus":
                return nn.Softplus(beta=softplus_beta)
            raise ValueError(f"Unsupported DivFreeFieldSimple activation: {activation}")

        # scalar function H: R^n -> R
        layers = []
        in_dim = n
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), make_activation()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        mixing = torch.randn(n, n)
        if self.learnable_mixing and self.normalize_mixing:
            if self.normalized_mixing_parameterization == "cayley":
                # Learn an orthogonal change of basis of a canonical skew matrix.
                # The resulting M stays skew-symmetric with spectral norm one.
                self.mixing_raw = nn.Parameter(1e-2 * mixing)
                canonical = torch.zeros(n, n)
                for index in range(0, n - 1, 2):
                    canonical[index, index + 1] = 1.0
                    canonical[index + 1, index] = -1.0
                self.register_buffer("canonical_mixing", canonical)
            else:
                # This begins from the same distribution as the fixed normalized
                # matrix while allowing all skew entries and singular values to
                # adapt. Normalization is recomputed differentiably.
                self.mixing_raw = nn.Parameter(mixing)
        elif self.learnable_mixing:
            self.mixing_raw = nn.Parameter(mixing)
        else:
            mixing = mixing - mixing.T
            if self.normalize_mixing:
                mixing = mixing / torch.linalg.matrix_norm(mixing, ord=2)
            self.register_buffer("M", mixing)

        if self.learnable_scale:
            initial_scale_tensor = torch.tensor(float(initial_scale))
            if self.bounded_scale_max is None:
                initial_raw = torch.log(torch.expm1(initial_scale_tensor))
            else:
                maximum = float(self.bounded_scale_max)
                if not 0.0 < float(initial_scale) < maximum:
                    raise ValueError("initial_scale must lie between zero and bounded_scale_max")
                initial_raw = torch.logit(initial_scale_tensor / maximum)
            self.scale_raw = nn.Parameter(initial_raw)
        else:
            self.scale = float(initial_scale)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def H(self, params, x):
        return self.current_scale() * functional_call(self.net, params, (x,)).squeeze(-1)

    def current_scale(self):
        if self.learnable_scale:
            if self.bounded_scale_max is not None:
                return float(self.bounded_scale_max) * torch.sigmoid(self.scale_raw)
            return F.softplus(self.scale_raw)
        return self.scale

    def mixing_matrix(self):
        if self.learnable_mixing and self.normalize_mixing:
            skew = self.mixing_raw - self.mixing_raw.T
            if self.normalized_mixing_parameterization == "normalized_skew":
                return skew / torch.linalg.matrix_norm(skew, ord=2).clamp_min(1e-12)
            identity = torch.eye(self.n, dtype=skew.dtype, device=skew.device)
            orthogonal = torch.linalg.solve(identity + skew, identity - skew)
            return orthogonal @ self.canonical_mixing @ orthogonal.T
        if self.learnable_mixing:
            return self.mixing_raw - self.mixing_raw.T
        return self.M

    def grad_H(self, params, x):
        return jacrev(self.H, argnums=1)(params, x)  # ∇H(x)

    def v_single(self, params, mixing, x):
        g = self.grad_H(params, x)      # shape (n,)
        return mixing @ g

    def forward(self, x):
        params = dict(self.net.named_parameters())
        mixing = self.mixing_matrix()
        return vmap(lambda xi: self.v_single(params, mixing, xi))(x)

class DivFreeFieldMultiPotential(nn.Module):
    """Sum of independent constant-skew Hamiltonian vector fields."""

    def __init__(
        self,
        n,
        hidden_dim=64,
        activation="leaky_relu",
        leaky_relu_slope=0.1,
        softplus_beta=5.0,
        num_hidden_layers=1,
        num_potentials=3,
        normalize_mixing=True,
        initial_scale=1.0,
        learnable_mixing=False,
        normalized_mixing_parameterization="cayley",
        learnable_gains=False,
        total_gain=13.0,
    ):
        super().__init__()
        if num_potentials < 1:
            raise ValueError("num_potentials must be at least one")
        if total_gain <= 0:
            raise ValueError("total_gain must be positive")
        self.learnable_gains = learnable_gains
        self.total_gain = float(total_gain)
        branch_scale = 1.0 if self.learnable_gains else initial_scale
        self.fields = nn.ModuleList(
            [
                DivFreeFieldSimple(
                    n,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    leaky_relu_slope=leaky_relu_slope,
                    softplus_beta=softplus_beta,
                    num_hidden_layers=num_hidden_layers,
                    learnable_mixing=learnable_mixing,
                    normalize_mixing=normalize_mixing,
                    learnable_scale=False,
                    initial_scale=branch_scale,
                    normalized_mixing_parameterization=normalized_mixing_parameterization,
                )
                for _ in range(num_potentials)
            ]
        )

        if self.learnable_gains:
            # Zero logits reproduce equal gains of total_gain / sqrt(K).
            self.gain_logits = nn.Parameter(torch.zeros(num_potentials))

    def current_gains(self):
        if not self.learnable_gains:
            return None
        probabilities = torch.softmax(self.gain_logits, dim=0)
        return self.total_gain * probabilities / torch.linalg.vector_norm(probabilities)

    def forward(self, x):
        outputs = [field(x) for field in self.fields]
        gains = self.current_gains()
        if gains is not None:
            outputs = [gain * output for gain, output in zip(gains, outputs)]
        return torch.stack(outputs, dim=0).sum(dim=0)


class DivFreeFieldSequential(nn.Module):
    """Piecewise-autonomous sequence of divergence-free scalar potentials."""

    uses_time = True

    def __init__(
        self,
        n,
        hidden_dim=64,
        activation="leaky_relu",
        leaky_relu_slope=0.1,
        softplus_beta=5.0,
        num_hidden_layers=1,
        num_stages=2,
        normalize_mixing=False,
        initial_scale=0.6,
    ):
        super().__init__()
        if num_stages < 1:
            raise ValueError("num_stages must be at least one")
        self.num_stages = num_stages
        self.fields = nn.ModuleList(
            [
                DivFreeFieldSimple(
                    n,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    leaky_relu_slope=leaky_relu_slope,
                    softplus_beta=softplus_beta,
                    num_hidden_layers=num_hidden_layers,
                    learnable_mixing=False,
                    normalize_mixing=normalize_mixing,
                    learnable_scale=False,
                    initial_scale=initial_scale,
                )
                for _ in range(num_stages)
            ]
        )

    def forward(self, t, x):
        scalar_time = float(t.reshape(-1)[0].detach().cpu())
        stage = min(int(scalar_time * self.num_stages), self.num_stages - 1)
        return self.fields[stage](x)


class DivFreeFieldSimple5(nn.Module):
    def __init__(self, n, hidden_dim = 64, num_non_zero=10):
        super().__init__()
        self.n = n
        self.scale = 1e-2
        self.hidden_dim = hidden_dim
        self.num_non_zero = num_non_zero

        # scalar function H: R^n -> R
        self.net1 = nn.Sequential(
            nn.Linear(n, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        # scalar function H: R^n -> R
        self.net2 = nn.Sequential(
            nn.Linear(n, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        # scalar function H: R^n -> R
        self.net3 = nn.Sequential(
            nn.Linear(n, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.nets = nn.ModuleList([self.net1, self.net2, self.net3])

        # num_matrices = 3
        # num_non_zero = self.num_non_zero  # per matrix

        # lower_tri_indices = [(i, j) for i in range(1, n) for j in range(i)]
        # assert num_matrices * num_non_zero <= len(lower_tri_indices)
        # all_chosen = random.sample(lower_tri_indices, num_matrices * num_non_zero)

        # Ms = []

        # for k in range(num_matrices):
        #     M = torch.zeros(n, n)

        #     # slice unique coordinates for this matrix
        #     chosen = all_chosen[k * num_non_zero : (k + 1) * num_non_zero]

        #     for i, j in chosen:
        #         value = torch.randn(1).item()
        #         M[i, j] = value
        #         M[j, i] = -value

        #     Ms.append(M)

        M1 = torch.randn(n, n)
        self.register_buffer("M", M1 - M1.T)
        M2 = torch.randn(n, n)
        self.register_buffer("M2", M2 - M2.T)
        M3 = torch.randn(n, n)
        self.register_buffer("M3", M3 - M3.T)

        Ms = [self.M, self.M2, self.M3]

        self.register_buffer("Ms", torch.stack(Ms))

        self.reset_parameters()

    def reset_parameters(self):
        for net in self.nets:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

    def H(self, params, x, i):
        return self.scale * functional_call(self.nets[i], params, (x,)).squeeze(-1)
        
    def grad_H(self, params, x, i):
            return jacrev(self.H, argnums=1)(params, x, i)  # ∇H(x)
       
    def v_single(self, params, x, i):

        g = self.grad_H(params, x, i)      # shape (n,)
        return self.Ms[i] @ g              # v = M ∇H

    def forward(self, x):
        params0 = dict(self.nets[0].named_parameters())
        params1 = dict(self.nets[1].named_parameters())
        params2 = dict(self.nets[2].named_parameters())

        def f(xi):
            return (
                self.v_single(params0, xi, 0)
                + self.v_single(params1, xi, 1)
                + self.v_single(params2, xi, 2)
            )
    
        if not self.training:
            with torch.no_grad():
                return vmap(f)(x)

        return vmap(f)(x)
