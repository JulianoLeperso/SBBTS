"""
Brownian Bridge sampling for SBBTS.

Implements Eq. (4.2) from the SBBTS paper for sampling from the conditional
law W|y_{t_i}, y_{t_{i+1}} of a Brownian bridge.
"""

import torch
from torch import Tensor


def brownian_bridge_mean(
    y_start: Tensor,
    y_end: Tensor,
    t: Tensor,
    t_i: float,
    t_i1: float,
) -> Tensor:
    """
    Compute the mean of the Brownian bridge at time t.

    Eq. (4.2) mean component:
        E[y_t | y_{t_i}, y_{t_{i+1}}] = ((t_{i+1} - t) / Δt_i) * y_{t_i}
                                      + ((t - t_i) / Δt_i) * y_{t_{i+1}}

    Args:
        y_start: Values at t_i, shape (batch, d) or (d,)
        y_end: Values at t_{i+1}, shape (batch, d) or (d,)
        t: Current time(s), shape (batch,) or scalar
        t_i: Start time of interval
        t_i1: End time of interval

    Returns:
        Mean at time t, same shape as y_start
    """
    dt = t_i1 - t_i
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=y_start.dtype, device=y_start.device)

    if t.dim() == 0:
        w_start = (t_i1 - t) / dt
        w_end = (t - t_i) / dt
    else:
        w_start = ((t_i1 - t) / dt).unsqueeze(-1)
        w_end = ((t - t_i) / dt).unsqueeze(-1)

    return w_start * y_start + w_end * y_end


def brownian_bridge_std(
    t: Tensor,
    t_i: float,
    t_i1: float,
) -> Tensor:
    """
    Compute the standard deviation of the Brownian bridge at time t.

    Eq. (4.2) variance component:
        σ²_t = (t - t_i)(t_{i+1} - t) / Δt_i

    Args:
        t: Current time(s), shape (batch,) or scalar
        t_i: Start time of interval
        t_i1: End time of interval

    Returns:
        Standard deviation at time t
    """
    dt = t_i1 - t_i
    var_t = (t - t_i) * (t_i1 - t) / dt
    return torch.sqrt(var_t)


def sample_brownian_bridge(
    y_start: Tensor,
    y_end: Tensor,
    t: Tensor,
    t_i: float,
    t_i1: float,
    generator: torch.Generator = None,
) -> Tensor:
    """
    Sample from the Brownian bridge W|y_{t_i}, y_{t_{i+1}}.

    Implements Eq. (4.2) from SBBTS paper:
        y_t = ((t_{i+1} - t) / Δt_i) * y_{t_i}
            + ((t - t_i) / Δt_i) * y_{t_{i+1}}
            + σ_t * Z
    where σ²_t = (t - t_i)(t_{i+1} - t) / Δt_i and Z ~ N(0, I_d)

    Args:
        y_start: Values at t_i, shape (batch, d)
        y_end: Values at t_{i+1}, shape (batch, d)
        t: Current time(s), shape (batch,) or scalar
        t_i: Start time of interval
        t_i1: End time of interval
        generator: Optional random generator for reproducibility

    Returns:
        Samples at time t, shape (batch, d)
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=y_start.dtype, device=y_start.device)

    mean = brownian_bridge_mean(y_start, y_end, t, t_i, t_i1)
    std = brownian_bridge_std(t, t_i, t_i1)

    if std.dim() == 0:
        std = std.unsqueeze(0)
    if std.dim() == 1 and mean.dim() == 2:
        std = std.unsqueeze(-1)

    z = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)

    return mean + std * z


def sample_brownian_bridge_batch(
    y_start: Tensor,
    y_end: Tensor,
    t_i: float,
    t_i1: float,
    n_samples: int = 1,
    generator: torch.Generator = None,
) -> tuple[Tensor, Tensor]:
    """
    Sample times uniformly and corresponding Brownian bridge values.

    Used in Algorithm 1 training loop: t ~ U([t_i, t_{i+1})).

    Args:
        y_start: Values at t_i, shape (batch, d)
        y_end: Values at t_{i+1}, shape (batch, d)
        t_i: Start time of interval
        t_i1: End time of interval
        n_samples: Number of time samples per batch element
        generator: Optional random generator

    Returns:
        Tuple of (t, y_t) where:
            t: Sampled times, shape (batch,) or (batch, n_samples)
            y_t: Bridge samples, shape (batch, d) or (batch, n_samples, d)
    """
    batch_size = y_start.shape[0]
    device = y_start.device
    dtype = y_start.dtype

    if n_samples == 1:
        t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        t = t_i + t * (t_i1 - t_i)
        y_t = sample_brownian_bridge(y_start, y_end, t, t_i, t_i1, generator)
    else:
        t = torch.rand(batch_size, n_samples, device=device, dtype=dtype, generator=generator)
        t = t_i + t * (t_i1 - t_i)
        y_start_exp = y_start.unsqueeze(1).expand(-1, n_samples, -1)
        y_end_exp = y_end.unsqueeze(1).expand(-1, n_samples, -1)
        y_t = sample_brownian_bridge(
            y_start_exp.reshape(-1, y_start.shape[-1]),
            y_end_exp.reshape(-1, y_end.shape[-1]),
            t.reshape(-1),
            t_i,
            t_i1,
            generator,
        ).reshape(batch_size, n_samples, -1)

    return t, y_t
