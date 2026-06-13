"""
Schrödinger-Bridge-Bass (SBB) solver for two-marginal problem.

Implements the base SBB problem from Section 2:
    SBB(μ_0, μ_T) = inf_{P ∈ P(μ_0, μ_T)} J(P)

where J(P) = E_P[∫_0^T ||α_t||² + β||σ_t - I_d||² dt]

This serves as the theoretical foundation for the SBBTS decomposition (Theorem 3.2).
"""

import torch
from torch import Tensor

from sbbts.transport.transport_map import validate_beta_condition


class SBBProblem:
    """
    Two-marginal Schrödinger-Bridge-Bass problem.

    From Eq. (2.1):
        J(P) = E_P[∫_0^T ||α_t||² + β||σ_t - I_d||² dt]

    The solution involves finding (h, ν, Y) satisfying:
        - h_t = h_T * N_{T-t} (backward heat equation)
        - ν_t = ν_0 * N_t (forward heat equation)
        - Y_t = (∇_y Φ_t)^{-1} with Φ_t(y) = |y|²/2 + (1/β) log h_t(y)

    Existence requires β * T > 1.
    """

    def __init__(
        self,
        beta: float,
        T: float = 1.0,
    ):
        """
        Args:
            beta: Regularization parameter β > 0
            T: Terminal time

        Raises:
            ValueError: If β * T ≤ 1 (existence condition violated)
        """
        if beta <= 0:
            raise ValueError(f"β must be positive, got {beta}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        validate_beta_condition(beta, T)

        self.beta = beta
        self.T = T

    @property
    def regime(self) -> str:
        """
        Determine the regime of the SBB problem.

        - β → ∞: Schrödinger Bridge (SB) regime, σ = I_d
        - β → 0: Bass martingale regime, α = 0
        - Moderate β: Interpolating regime
        """
        if self.beta > 100:
            return "schrodinger_bridge"
        elif self.beta < 1:
            return "bass_martingale"
        else:
            return "interpolating"

    def compute_optimal_volatility_approx(
        self,
        x: Tensor,
        score: Tensor,
    ) -> Tensor:
        """
        Compute approximate optimal volatility σ*_t.

        For large β, from Section 2:
            σ*_t = D²_y Φ_t(Y_t(X_t)) ≈ I_d + O(1/β)

        In practice for large β regime, volatility is close to identity.

        Args:
            x: State tensor, shape (batch, d)
            score: Score ∇_y log h, shape (batch, d)

        Returns:
            Approximate volatility, shape (batch, d, d)
        """
        batch_size, d = x.shape
        device = x.device
        dtype = x.dtype

        identity = torch.eye(d, device=device, dtype=dtype)
        vol = identity.unsqueeze(0).expand(batch_size, -1, -1).clone()

        return vol


class DSBDynamics:
    """
    Diffusion Schrödinger Bridge dynamics.

    From Section 2, the process Y_t = Y_t(X_t) evolves as:
        dY_t = ∇_y log h_t(Y_t) dt + dW_t  under P^SBB

    This is used for sampling new trajectories after training.
    """

    def __init__(
        self,
        score_fn: callable,
        dt: float = 0.02,
    ):
        """
        Args:
            score_fn: Score function s_θ ≈ ∇_y log h
            dt: Time step for Euler-Maruyama discretization
        """
        self.score_fn = score_fn
        self.dt = dt

    def step(
        self,
        y: Tensor,
        t: float,
        context: Tensor = None,
        generator: torch.Generator = None,
    ) -> Tensor:
        """
        Perform one Euler-Maruyama step of the DSB dynamics.

        dY_t = s_θ(t, Y_t, context) dt + dW_t

        Args:
            y: Current state Y_t, shape (batch, d)
            t: Current time
            context: Optional context vector for score function
            generator: Optional random generator

        Returns:
            Next state Y_{t+dt}, shape (batch, d)
        """
        t_tensor = torch.tensor(t, device=y.device, dtype=y.dtype)
        drift = self.score_fn(t_tensor.expand(y.shape[0]), y, context)

        noise = torch.randn_like(y, generator=generator) * (self.dt**0.5)

        return y + drift * self.dt + noise

    def simulate(
        self,
        y0: Tensor,
        t_start: float,
        t_end: float,
        context: Tensor = None,
        generator: torch.Generator = None,
    ) -> Tensor:
        """
        Simulate DSB dynamics from t_start to t_end.

        Args:
            y0: Initial state, shape (batch, d)
            t_start: Start time
            t_end: End time
            context: Optional context vector
            generator: Optional random generator

        Returns:
            Final state Y_{t_end}, shape (batch, d)
        """
        n_steps = max(1, int((t_end - t_start) / self.dt))
        actual_dt = (t_end - t_start) / n_steps

        y = y0.clone()
        t = t_start

        for _ in range(n_steps):
            t_tensor = torch.tensor(t, device=y.device, dtype=y.dtype)
            drift = self.score_fn(t_tensor.expand(y.shape[0]), y, context)
            noise = torch.randn_like(y, generator=generator) * (actual_dt**0.5)
            y = y + drift * actual_dt + noise
            t = t + actual_dt

        return y
