"""
Heston model benchmark for SBBTS.

Implements the Heston stochastic volatility model from Section 5.1:
    dX_t = r X_t dt + √v_t X_t dW^X_t
    dv_t = κ(θ - v_t) dt + ξ √v_t dW^v_t

where Corr(W^X_t, W^v_t) = ρ

Parameters from Table 3:
    κ ∈ [0.5, 4]      (mean reversion speed)
    θ ∈ [0.5, 1.5]    (long-term variance)
    ξ ∈ [0.1, 0.9]    (vol of vol)
    ρ ∈ [-0.9, 0.9]   (correlation)
    r ∈ [0.01, 0.1]   (risk-free rate)
"""

from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import minimize

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(func):  # no-op decorator when numba unavailable
        return func


@dataclass
class HestonParams:
    """Heston model parameters."""
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    xi: float     # Vol of vol
    rho: float    # Correlation
    r: float      # Risk-free rate
    v0: float = None  # Initial variance (defaults to theta)

    def __post_init__(self):
        if self.v0 is None:
            self.v0 = self.theta

    def to_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "r": self.r,
            "v0": self.v0,
        }


def sample_heston_params(
    n_samples: int = 1,
    seed: int = None,
) -> list:
    """
    Sample Heston parameters from ranges in Table 3.

    Args:
        n_samples: Number of parameter sets to sample
        seed: Random seed

    Returns:
        List of HestonParams
    """
    if seed is not None:
        np.random.seed(seed)

    params_list = []
    for _ in range(n_samples):
        params = HestonParams(
            kappa=np.random.uniform(0.5, 4.0),
            theta=np.random.uniform(0.5, 1.5),
            xi=np.random.uniform(0.1, 0.9),
            rho=np.random.uniform(-0.9, 0.9),
            r=np.random.uniform(0.01, 0.1),
        )
        params_list.append(params)

    return params_list if n_samples > 1 else params_list[0]


def simulate_heston(
    params: HestonParams,
    n_paths: int,
    n_steps: int,
    dt: float = 1/252,
    S0: float = 100.0,
    return_variance: bool = True,
    seed: int = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate Heston model paths using Euler discretization.

    dX_t = r X_t dt + √v_t X_t dW^X_t
    dv_t = κ(θ - v_t) dt + ξ √v_t dW^v_t

    Args:
        params: Heston parameters
        n_paths: Number of paths
        n_steps: Number of time steps
        dt: Time step (1/252 for daily)
        S0: Initial price
        return_variance: Also return variance paths

    Returns:
        If return_variance: (prices, variances), each shape (n_paths, n_steps+1)
        Else: prices, shape (n_paths, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    kappa, theta, xi, rho, r, v0 = (
        params.kappa, params.theta, params.xi, params.rho, params.r, params.v0
    )

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)

        W_X = Z1
        W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v_t = np.maximum(v[:, t], 0)
        sqrt_v = np.sqrt(v_t)

        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_t) * dt + sqrt_v * sqrt_dt * W_X
        )

        v[:, t + 1] = v_t + kappa * (theta - v_t) * dt + xi * sqrt_v * sqrt_dt * W_v
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)

    if return_variance:
        return S, v
    return S


def simulate_heston_log_returns(
    params: HestonParams,
    n_paths: int,
    n_steps: int,
    dt: float = 1/252,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate Heston model and return log returns.

    Used for SBBTS training as in Section 5.1.

    Args:
        params: Heston parameters
        n_paths: Number of paths
        n_steps: Number of time steps (returns will have n_steps+1 points)
        dt: Time step
        seed: Random seed

    Returns:
        Log returns, shape (n_paths, n_steps+1, 2)
        First column: log(S_t / S_0)
        Second column: v_t (variance)
    """
    S, v = simulate_heston(params, n_paths, n_steps, dt, S0=100.0, seed=seed)

    log_S = np.log(S / S[:, 0:1])

    trajectory = np.stack([log_S, v], axis=-1)

    return trajectory


def generate_heston_dataset(
    n_trajectories: int = 5000,
    trajectory_length: int = 252,
    heterogeneous: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, list]:
    """
    Generate Heston dataset as in Section 5.1.

    From the paper: "each parameter vector is independently sampled from
    a prescribed range, so that the training dataset consists of Heston
    time series generated under heterogeneous parameter configurations."

    Args:
        n_trajectories: Number of trajectories (Section 5.1: 5000)
        trajectory_length: Length of each trajectory (Section 5.1: 252)
        heterogeneous: Sample different params per trajectory (True) or use same
        seed: Random seed

    Returns:
        Tuple of:
            - trajectories: shape (n_trajectories, trajectory_length+1, 2)
            - params_list: list of HestonParams used for each trajectory
    """
    np.random.seed(seed)

    if heterogeneous:
        params_list = sample_heston_params(n_trajectories, seed=seed)
    else:
        params_list = [sample_heston_params(1, seed=seed)] * n_trajectories

    trajectories = np.zeros((n_trajectories, trajectory_length + 1, 2))

    for i, params in enumerate(params_list):
        traj = simulate_heston_log_returns(
            params, n_paths=1, n_steps=trajectory_length,
            seed=seed + i if seed else None
        )
        trajectories[i] = traj[0]

    return trajectories, params_list


@_njit
def _heston_variance_nll(v: np.ndarray, kappa: float, theta: float, xi: float, dt: float) -> float:
    """
    Negative log-likelihood of the Heston variance process (OU approximation).
    JIT-compiled with Numba when available for fast repeated MLE calls.
    """
    nll = 0.0
    for t in range(len(v) - 1):
        v_t = max(v[t], 1e-10)
        mu_t = v_t + kappa * (theta - v_t) * dt
        sigma_t = xi * (v_t ** 0.5) * (dt ** 0.5)
        if sigma_t < 1e-12:
            continue
        diff = v[t + 1] - mu_t
        nll += 0.5 * (diff / sigma_t) ** 2 + 0.5 * np.log(2.0 * np.pi * sigma_t ** 2)
    return nll


def estimate_heston_mle(
    trajectory: np.ndarray,
    dt: float = 1/252,
) -> HestonParams:
    """
    Estimate Heston parameters using Maximum Likelihood Estimation.

    From Section 5.1: "the Heston parameters are estimated on each
    generated sample using a maximum likelihood approach"

    This is a simplified MLE based on moment matching for the variance process.

    Args:
        trajectory: Single trajectory, shape (T, 2) with [log_S, v]
        dt: Time step

    Returns:
        Estimated HestonParams
    """
    log_S = trajectory[:, 0]
    v = trajectory[:, 1]

    T = len(v)

    mean_v = np.mean(v)
    var_v = np.var(v)

    theta_est = mean_v

    dv = np.diff(v)
    v_lag = v[:-1]
    v_lag_safe = np.maximum(v_lag, 1e-8)

    kappa_est = -np.mean(dv / (theta_est - v_lag_safe + 1e-8)) / dt
    kappa_est = np.clip(kappa_est, 0.1, 10.0)

    residuals = dv - kappa_est * (theta_est - v_lag) * dt
    xi_est = np.std(residuals / np.sqrt(v_lag_safe * dt + 1e-8))
    xi_est = np.clip(xi_est, 0.01, 2.0)

    log_returns = np.diff(log_S)
    dv_norm = dv / np.sqrt(v_lag_safe * dt + 1e-8)
    log_ret_norm = log_returns / np.sqrt(v_lag_safe * dt + 1e-8)

    if np.std(dv_norm) > 0 and np.std(log_ret_norm) > 0:
        rho_est = np.corrcoef(dv_norm, log_ret_norm)[0, 1]
    else:
        rho_est = 0.0
    rho_est = np.clip(rho_est, -0.99, 0.99)

    r_est = np.mean(log_returns) / dt + 0.5 * mean_v
    r_est = np.clip(r_est, 0.0, 0.5)

    return HestonParams(
        kappa=kappa_est,
        theta=theta_est,
        xi=xi_est,
        rho=rho_est,
        r=r_est,
        v0=v[0],
    )


def evaluate_parameter_recovery(
    true_params_list: list,
    estimated_params_list: list,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate parameter recovery accuracy.

    Computes statistics of estimation errors as in Figure 2.

    Args:
        true_params_list: List of true HestonParams
        estimated_params_list: List of estimated HestonParams

    Returns:
        Dictionary with mean and std of errors for each parameter
    """
    param_names = ["kappa", "theta", "xi", "rho", "r"]
    results = {}

    for name in param_names:
        true_vals = np.array([getattr(p, name) for p in true_params_list])
        est_vals = np.array([getattr(p, name) for p in estimated_params_list])

        errors = est_vals - true_vals

        results[name] = {
            "true_mean": np.mean(true_vals),
            "true_std": np.std(true_vals),
            "est_mean": np.mean(est_vals),
            "est_std": np.std(est_vals),
            "bias": np.mean(errors),
            "rmse": np.sqrt(np.mean(errors**2)),
            "mae": np.mean(np.abs(errors)),
        }

    return results
