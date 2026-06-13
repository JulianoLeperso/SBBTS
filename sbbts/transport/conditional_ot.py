"""
Conditional Optimal Transport for SBBTS.

Implements V_i(x_{0:i}) from Theorem 3.2:
    V_i(x_{0:i}) = SBB(δ_{x_i}, μ_{i+1|0:i}(·|x_{0:i}))

This is the conditional transport problem that transports from Dirac δ_{x_i}
to the conditional distribution μ_{i+1|0:i} on each interval [t_i, t_{i+1}].
"""

import torch
from torch import Tensor

from sbbts.transport.brownian_bridge import sample_brownian_bridge_batch
from sbbts.transport.transport_map import x_to_y, y_to_x


def compute_conditional_transport(
    x_ti: Tensor,
    x_ti1: Tensor,
    score_fn: callable,
    beta: float,
    t_i: float,
    t_i1: float,
    context: Tensor = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute Y-space transport for interval [t_i, t_{i+1}].

    From Algorithm 1, lines 5-6:
        Y^b_{t_i} = X^b_{t_i} - (1/β) s^k_θ(t_i, X^b_{t_i}, Φ^k_θ(X^b_{t_{0:i}}))
        Y^b_{t_{i+1}} = X^b_{t_{i+1}} - (1/β) s^k_θ(t_{i+1}, X^b_{t_{i+1}}, Φ^k_θ(X^b_{t_{0:i}}))

    Note: Both Y_{t_i} and Y_{t_{i+1}} use the SAME context Φθ(X_{t_{0:i}}).

    Args:
        x_ti: States at t_i, shape (batch, d)
        x_ti1: States at t_{i+1}, shape (batch, d)
        score_fn: Score function (t, x, context) -> score
        beta: Regularization parameter β
        t_i: Start time of interval
        t_i1: End time of interval
        context: Context vector c_i = Φθ(X_{t_{0:i}}), shape (batch, d_model)

    Returns:
        Tuple of:
            - y_ti: Transported states at t_i, shape (batch, d)
            - y_ti1: Transported states at t_{i+1}, shape (batch, d)
            - score_ti: Score at t_i, shape (batch, d)
            - score_ti1: Score at t_{i+1}, shape (batch, d)
    """
    t_ti = torch.tensor(t_i, device=x_ti.device, dtype=x_ti.dtype)
    t_ti1 = torch.tensor(t_i1, device=x_ti.device, dtype=x_ti.dtype)

    score_ti = score_fn(t_ti.expand(x_ti.shape[0]), x_ti, context)
    y_ti = x_to_y(x_ti, score_ti, beta)

    score_ti1 = score_fn(t_ti1.expand(x_ti1.shape[0]), x_ti1, context)
    y_ti1 = x_to_y(x_ti1, score_ti1, beta)

    return y_ti, y_ti1, score_ti, score_ti1


def sample_training_points(
    y_ti: Tensor,
    y_ti1: Tensor,
    t_i: float,
    t_i1: float,
    generator: torch.Generator = None,
) -> tuple[Tensor, Tensor]:
    """
    Sample training points from Brownian bridge for loss computation.

    From Algorithm 1, line 7:
        Sample {Y^b_t ~ W|Y^b_{t_i}, Y^b_{t_{i+1}}}

    And the loss target from Eq. (4.1):
        (Y_{t_{i+1}} - Y_t) / (t_{i+1} - t)

    Args:
        y_ti: Y-space states at t_i, shape (batch, d)
        y_ti1: Y-space states at t_{i+1}, shape (batch, d)
        t_i: Start time of interval
        t_i1: End time of interval
        generator: Optional random generator

    Returns:
        Tuple of:
            - t: Sampled times, shape (batch,)
            - y_t: Brownian bridge samples, shape (batch, d)
    """
    return sample_brownian_bridge_batch(y_ti, y_ti1, t_i, t_i1, n_samples=1, generator=generator)


def compute_score_target(
    y_ti1: Tensor,
    y_t: Tensor,
    t: Tensor,
    t_i1: float,
) -> Tensor:
    """
    Compute the target for score matching loss.

    From Eq. (4.1):
        target = (Y_{t_{i+1}} - Y_t) / (t_{i+1} - t)

    This is the conditional expectation of the drift that the score network
    should learn.

    Args:
        y_ti1: Y-space states at t_{i+1}, shape (batch, d)
        y_t: Y-space states at sampled time t, shape (batch, d)
        t: Sampled times, shape (batch,)
        t_i1: End time of interval

    Returns:
        Score target, shape (batch, d)
    """
    dt = (t_i1 - t).unsqueeze(-1)
    return (y_ti1 - y_t) / dt


def compute_interval_loss(
    score_pred: Tensor,
    score_target: Tensor,
) -> Tensor:
    """
    Compute the score matching loss for one interval.

    From Eq. (4.1):
        L_i = ||s_θ(t, Y_t, Φθ(Y_{t_{0:i}})) - (Y_{t_{i+1}} - Y_t)/(t_{i+1} - t)||²_2

    Args:
        score_pred: Predicted score, shape (batch, d)
        score_target: Target score, shape (batch, d)

    Returns:
        Mean squared error loss, scalar
    """
    return ((score_pred - score_target) ** 2).sum(dim=-1).mean()


class ConditionalOTSolver:
    """
    Solver for the conditional optimal transport problem V_i(x_{0:i}).

    Theorem 3.2: SBBTS decomposes into solving SBB(δ_{x_i}, μ_{i+1|0:i})
    on each interval, conditioned on the past trajectory.
    """

    def __init__(
        self,
        beta: float,
        t_tilde_offset: float = 0.01,
    ):
        """
        Args:
            beta: Regularization parameter β
            t_tilde_offset: Offset ξ for evaluating at t̃_{i+1} = t_{i+1} - ξ
                           (see note after Algorithm 1 in the paper)
        """
        self.beta = beta
        self.t_tilde_offset = t_tilde_offset

    def compute_loss_for_interval(
        self,
        x_ti: Tensor,
        x_ti1: Tensor,
        score_net: callable,
        t_i: float,
        t_i1: float,
        context: Tensor,
        generator: torch.Generator = None,
    ) -> Tensor:
        """
        Compute the training loss for one interval [t_i, t_{i+1}].

        Implements the core loop of Algorithm 1 (lines 5-8) for a single interval.

        Args:
            x_ti: States at t_i, shape (batch, d)
            x_ti1: States at t_{i+1}, shape (batch, d)
            score_net: Score network (t, x, context) -> score
            t_i: Start time of interval
            t_i1: End time of interval (use t̃_{i+1} = t_{i+1} - ξ)
            context: Context vector c_i, shape (batch, d_model)
            generator: Optional random generator

        Returns:
            Loss for this interval, scalar
        """
        t_i1_tilde = t_i1 - self.t_tilde_offset

        y_ti, y_ti1, _, _ = compute_conditional_transport(
            x_ti, x_ti1, score_net, self.beta, t_i, t_i1_tilde, context
        )

        t, y_t = sample_training_points(y_ti, y_ti1, t_i, t_i1_tilde, generator)

        score_target = compute_score_target(y_ti1, y_t, t, t_i1_tilde)

        score_pred = score_net(t, y_t, context)

        return compute_interval_loss(score_pred, score_target)

    def compute_total_loss(
        self,
        trajectory: Tensor,
        score_net: callable,
        time_points: Tensor,
        contexts: Tensor,
        generator: torch.Generator = None,
    ) -> Tensor:
        """
        Compute the total training loss over all intervals.

        From Eq. (4.1):
            L(θ) = (1/N) Σ_{i=0}^{N-1} L_i

        Args:
            trajectory: Full trajectory X_{t_0:t_N}, shape (batch, N+1, d)
            score_net: Score network
            time_points: Observation times [t_0, ..., t_N], shape (N+1,)
            contexts: All context vectors, shape (batch, N, d_model)
            generator: Optional random generator

        Returns:
            Total loss, scalar
        """
        batch_size, n_points, d = trajectory.shape
        n_intervals = n_points - 1

        total_loss = 0.0
        for i in range(n_intervals):
            x_ti = trajectory[:, i, :]
            x_ti1 = trajectory[:, i + 1, :]
            t_i = time_points[i].item()
            t_i1 = time_points[i + 1].item()
            context = contexts[:, i, :]

            interval_loss = self.compute_loss_for_interval(
                x_ti, x_ti1, score_net, t_i, t_i1, context, generator
            )
            total_loss = total_loss + interval_loss

        return total_loss / n_intervals
