"""
Sampling utilities for SBBTS.

Implements Euler-Maruyama discretization for SDE simulation (Section C, Table 2: N_π = 50).
"""

import math
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
import numpy as np


def euler_maruyama_step(
    x: Tensor,
    t: float,
    dt: float,
    drift_fn: Callable[[Tensor, float], Tensor],
    diffusion_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
    generator: torch.Generator = None,
) -> Tensor:
    """
    Single Euler-Maruyama step for SDE simulation.

    For SDE: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t

    Args:
        x: Current state, shape (batch, d)
        t: Current time
        dt: Time step
        drift_fn: Drift function μ(x, t)
        diffusion_fn: Diffusion function σ(x, t), defaults to I_d
        generator: Optional random generator

    Returns:
        Next state X_{t+dt}, shape (batch, d)
    """
    drift = drift_fn(x, t)

    noise = torch.randn_like(x, generator=generator)

    if diffusion_fn is not None:
        diffusion = diffusion_fn(x, t)
        if diffusion.dim() == 3:
            noise = torch.einsum("bdi,bi->bd", diffusion, noise)
        else:
            noise = diffusion * noise

    noise = noise * math.sqrt(dt)

    return x + drift * dt + noise


def euler_maruyama_simulate(
    x0: Tensor,
    t_start: float,
    t_end: float,
    n_steps: int,
    drift_fn: Callable[[Tensor, float], Tensor],
    diffusion_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
    return_trajectory: bool = False,
    generator: torch.Generator = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Simulate SDE using Euler-Maruyama scheme.

    From Table 2: N_π = 50 steps for DSB simulation.

    Args:
        x0: Initial state, shape (batch, d)
        t_start: Start time
        t_end: End time
        n_steps: Number of discretization steps (N_π)
        drift_fn: Drift function μ(x, t)
        diffusion_fn: Optional diffusion function σ(x, t)
        return_trajectory: Return full trajectory or just final state
        generator: Optional random generator

    Returns:
        If return_trajectory: (trajectory, times) where trajectory is (batch, n_steps+1, d)
        Else: Final state X_{t_end}, shape (batch, d)
    """
    dt = (t_end - t_start) / n_steps

    if return_trajectory:
        batch_size, d = x0.shape
        trajectory = torch.zeros(batch_size, n_steps + 1, d, device=x0.device, dtype=x0.dtype)
        times = torch.linspace(t_start, t_end, n_steps + 1, device=x0.device)
        trajectory[:, 0, :] = x0

        x = x0.clone()
        for i in range(n_steps):
            t = t_start + i * dt
            x = euler_maruyama_step(x, t, dt, drift_fn, diffusion_fn, generator)
            trajectory[:, i + 1, :] = x

        return trajectory, times
    else:
        x = x0.clone()
        for i in range(n_steps):
            t = t_start + i * dt
            x = euler_maruyama_step(x, t, dt, drift_fn, diffusion_fn, generator)
        return x


def generate_brownian_motion(
    n_samples: int,
    n_steps: int,
    d: int,
    dt: float = 1.0,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> Tensor:
    """
    Generate standard Brownian motion paths.

    Args:
        n_samples: Number of sample paths
        n_steps: Number of time steps
        d: Dimension
        dt: Time step size
        device: Computation device
        generator: Optional random generator

    Returns:
        Brownian paths, shape (n_samples, n_steps+1, d)
    """
    if device is None:
        device = torch.device("cpu")

    increments = torch.randn(n_samples, n_steps, d, device=device, generator=generator)
    increments = increments * math.sqrt(dt)

    paths = torch.zeros(n_samples, n_steps + 1, d, device=device)
    paths[:, 1:, :] = torch.cumsum(increments, dim=1)

    return paths


def generate_gbm(
    n_samples: int,
    n_steps: int,
    d: int = 1,
    mu: float = 0.05,
    sigma: float = 0.2,
    S0: float = 100.0,
    dt: float = 1/252,
    return_log_prices: bool = True,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> Tensor:
    """
    Generate Geometric Brownian Motion paths.

    dS_t = μ S_t dt + σ S_t dW_t

    Solution: S_t = S_0 exp((μ - σ²/2)t + σ W_t)

    Args:
        n_samples: Number of sample paths
        n_steps: Number of time steps
        d: Dimension (multiple independent GBMs)
        mu: Drift parameter
        sigma: Volatility parameter
        S0: Initial price
        dt: Time step
        return_log_prices: Return log prices (True) or prices (False)
        device: Computation device
        generator: Optional random generator

    Returns:
        Paths, shape (n_samples, n_steps+1, d)
    """
    if device is None:
        device = torch.device("cpu")

    W = generate_brownian_motion(n_samples, n_steps, d, dt, device, generator)

    times = torch.arange(n_steps + 1, device=device).float() * dt
    times = times.unsqueeze(0).unsqueeze(-1)

    log_S = math.log(S0) + (mu - 0.5 * sigma**2) * times + sigma * W

    if return_log_prices:
        return log_S
    else:
        return torch.exp(log_S)


class SDESampler:
    """
    General-purpose SDE sampler using Euler-Maruyama.

    Useful for generating reference trajectories and testing.
    """

    def __init__(
        self,
        drift_fn: Callable[[Tensor, float], Tensor],
        diffusion_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
        n_steps: int = 50,
    ):
        """
        Args:
            drift_fn: Drift function μ(x, t)
            diffusion_fn: Diffusion function σ(x, t), defaults to identity
            n_steps: Number of Euler-Maruyama steps (Table 2: N_π = 50)
        """
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.n_steps = n_steps

    def sample(
        self,
        x0: Tensor,
        t_start: float,
        t_end: float,
        return_trajectory: bool = False,
        generator: torch.Generator = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Sample from the SDE.

        Args:
            x0: Initial state
            t_start: Start time
            t_end: End time
            return_trajectory: Return full trajectory
            generator: Optional random generator

        Returns:
            Samples from the SDE
        """
        return euler_maruyama_simulate(
            x0, t_start, t_end, self.n_steps,
            self.drift_fn, self.diffusion_fn,
            return_trajectory, generator
        )
