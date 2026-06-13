"""
Rough volatility models for SBBTS benchmarks.

Implements the rough Heston model using fractional Brownian motion (H < 0.5).
Empirically, equity volatility surfaces show H ≈ 0.1 (much rougher than BM).

Models:
    dX_t = r*X_t*dt + sqrt(V_t)*X_t*dW_t
    dV_t = κ(θ - V_t)dt + ν*sqrt(V_t)*dW^H_t,  Corr(W, W^H) = ρ

References:
    Gatheral et al. (2018), "Volatility is rough"
    El Euch & Rosenbaum (2019), "The characteristic function of rough Heston"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RoughHestonParams:
    """Parameters for the rough Heston model."""

    H: float = 0.1  # Hurst exponent — roughness; H < 0.5 for rough volatility
    kappa: float = 1.0  # Mean reversion speed
    theta: float = 0.04  # Long-term variance
    nu: float = 0.3  # Vol-of-vol
    rho: float = -0.7  # Spot-vol correlation (typically negative for equity)
    r: float = 0.02  # Risk-free rate
    V0: Optional[float] = None  # Initial variance (defaults to theta)

    def __post_init__(self):
        if self.V0 is None:
            self.V0 = self.theta


def simulate_fbm_riemann_liouville(
    n_paths: int,
    n_steps: int,
    H: float,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate fractional BM via Riemann-Liouville representation.

    W^H_t ≈ C_H * Σ_{k<t} (t - k)^{H - 0.5} * ΔW_k

    O(T²) but exact in distribution for large T.

    Args:
        n_paths: Number of sample paths
        n_steps: Number of time steps
        H: Hurst exponent
        dt: Time step
        seed: Random seed

    Returns:
        fBM paths, shape (n_paths, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    C_H = np.sqrt(2 * H) / (H + 0.5)
    Z = np.random.randn(n_paths, n_steps) * np.sqrt(dt)  # i.i.d. BM increments

    W_H = np.zeros((n_paths, n_steps + 1))
    for t in range(1, n_steps + 1):
        k = np.arange(t)
        weights = (t - k) ** (H - 0.5)
        W_H[:, t] = C_H * (Z[:, :t] * weights[::-1]).sum(axis=1)

    return W_H


def simulate_rough_heston(
    params: RoughHestonParams,
    n_paths: int,
    n_steps: int,
    dt: float = 1 / 252,
    S0: float = 100.0,
    return_variance: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate rough Heston model via Euler–Maruyama.

    Args:
        params: RoughHestonParams
        n_paths: Number of paths
        n_steps: Time steps
        dt: Time step (1/252 = daily)
        S0: Initial price
        return_variance: Also return variance process
        seed: Random seed

    Returns:
        (S, V) each (n_paths, n_steps+1), or just S if return_variance=False
    """
    if seed is not None:
        np.random.seed(seed)

    # Independent BM increments for price
    W_S = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

    # fBM increments for variance (correlated with W_S via ρ)
    W_H = simulate_fbm_riemann_liouville(n_paths, n_steps, params.H, dt)
    dW_H = np.diff(W_H, axis=1)  # (n_paths, n_steps)

    # Correlate: dW_vol = ρ*dW_S + sqrt(1-ρ²)*dW_H
    dW_vol = params.rho * W_S + np.sqrt(max(1 - params.rho**2, 0.0)) * dW_H

    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    V[:, 0] = params.V0

    for t in range(n_steps):
        V_t = np.maximum(V[:, t], 0.0)
        sv = np.sqrt(V_t)

        S[:, t + 1] = S[:, t] * np.exp((params.r - 0.5 * V_t) * dt + sv * W_S[:, t])
        V[:, t + 1] = np.maximum(
            V_t + params.kappa * (params.theta - V_t) * dt + params.nu * sv * dW_vol[:, t],
            0.0,
        )

    if return_variance:
        return S, V
    return S, None


def generate_rough_heston_dataset(
    n_trajectories: int = 2000,
    trajectory_length: int = 252,
    H: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, list]:
    """
    Generate heterogeneous rough Heston dataset for SBBTS training.

    Returns trajectories of [log-return, variance] with randomized parameters.

    Args:
        n_trajectories: Number of paths
        trajectory_length: Steps per path (252 = 1 trading year)
        H: Hurst exponent (0.1 = empirically calibrated rough vol)
        seed: Random seed

    Returns:
        (trajectories, params_list) where trajectories is (N, T+1, 2)
    """
    np.random.seed(seed)

    trajectories = np.zeros((n_trajectories, trajectory_length + 1, 2))
    params_list = []

    for i in range(n_trajectories):
        p = RoughHestonParams(
            H=H,
            kappa=np.random.uniform(0.5, 4.0),
            theta=np.random.uniform(0.01, 0.25),
            nu=np.random.uniform(0.1, 0.9),
            rho=np.random.uniform(-0.9, -0.1),
        )
        params_list.append(p)

        S, V = simulate_rough_heston(p, n_paths=1, n_steps=trajectory_length, seed=seed + i)
        log_S = np.log(S[0] / S[0, 0])
        trajectories[i] = np.stack([log_S, V[0]], axis=-1)

    return trajectories, params_list


def sample_rough_params(
    n: int = 1,
    H_range: Tuple[float, float] = (0.05, 0.45),
    seed: Optional[int] = None,
) -> list:
    """Sample random rough Heston parameters."""
    if seed is not None:
        np.random.seed(seed)
    params = [
        RoughHestonParams(
            H=np.random.uniform(*H_range),
            kappa=np.random.uniform(0.5, 4.0),
            theta=np.random.uniform(0.01, 0.25),
            nu=np.random.uniform(0.1, 0.9),
            rho=np.random.uniform(-0.9, -0.1),
        )
        for _ in range(n)
    ]
    return params if n > 1 else params[0]
