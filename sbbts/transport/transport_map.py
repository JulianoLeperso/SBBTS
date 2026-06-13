"""
Transport map Y <-> X for SBBTS.

Implements the large-β approximation from Section 4:
    Y_t(x) ≈ x - (1/β) ∇_y log h_t(x)

In our parametrization, s_θ ≈ ∇_y log h, so:
    Y = X - (1/β) s_θ(t, X, context)
    X = Y + (1/β) s_θ(t, Y, context)  (inverse, approximate)
"""

import torch
from torch import Tensor


def x_to_y(
    x: Tensor,
    score: Tensor,
    beta: float,
) -> Tensor:
    """
    Apply forward transport map: X -> Y.

    Large-β approximation from Section 4:
        Y_t(x) = x - (1/β) ∇_y log h_t(Y_t(x))
               ≈ x - (1/β) ∇_y log h_t(x)   [for large β]

    In our parametrization where s_θ ≈ ∇_y log h:
        Y = X - (1/β) s_θ(t, X, context)

    Args:
        x: Input state, shape (batch, d) or (d,)
        score: Score network output s_θ, same shape as x
        beta: Regularization parameter β > 0

    Returns:
        Transported state Y, same shape as x
    """
    return x - score / beta


def y_to_x(
    y: Tensor,
    score: Tensor,
    beta: float,
) -> Tensor:
    """
    Apply inverse transport map: Y -> X.

    From Section 2.3 of the paper:
        X_t = Y_t^{-1}(Y_t) = Y_t + (1/β) ∇_y log h_t(Y_t)

    In our parametrization:
        X = Y + (1/β) s_θ(t, Y, context)

    Note: This is exact for the inverse map, not an approximation.

    Args:
        y: Input state in Y-space, shape (batch, d) or (d,)
        score: Score network output s_θ, same shape as y
        beta: Regularization parameter β > 0

    Returns:
        Recovered state X, same shape as y
    """
    return y + score / beta


def validate_beta_condition(
    beta: float,
    dt: float,
    interval_idx: int = None,
) -> None:
    """
    Validate the condition β * Δt > 1 from Theorem 3.2.

    This condition is necessary for the existence of the SBB solution
    on each interval [t_i, t_{i+1}].

    Args:
        beta: Regularization parameter β
        dt: Time interval Δt_i = t_{i+1} - t_i
        interval_idx: Optional interval index for error message

    Raises:
        ValueError: If β * Δt ≤ 1
    """
    if beta * dt <= 1:
        msg = f"Theorem 3.2 condition violated: β * Δt = {beta * dt:.4f} ≤ 1"
        if interval_idx is not None:
            msg = f"Interval {interval_idx}: " + msg
        msg += f"\nRequire β > 1/Δt = {1/dt:.4f} (current β = {beta})"
        raise ValueError(msg)


def compute_transport_map_jacobian(
    x: Tensor,
    score_fn: callable,
    beta: float,
    t: float,
    context: Tensor = None,
    eps: float = 1e-4,
) -> Tensor:
    """
    Numerically estimate the Jacobian of the transport map Y = X - (1/β) s_θ.

    The Jacobian D_x Y = I - (1/β) D_x s_θ appears in the optimal volatility:
        σ*_t = D²_y Φ_t(Y_t(X_t))

    For large β, this is approximately the identity.

    Args:
        x: Input state, shape (batch, d)
        score_fn: Function (t, x, context) -> score
        beta: Regularization parameter
        t: Current time
        context: Optional context tensor
        eps: Finite difference step size

    Returns:
        Jacobian matrix, shape (batch, d, d)
    """
    batch_size, d = x.shape
    device = x.device
    dtype = x.dtype

    jacobian = torch.zeros(batch_size, d, d, device=device, dtype=dtype)

    for i in range(d):
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus[:, i] += eps
        x_minus[:, i] -= eps

        if context is not None:
            s_plus = score_fn(t, x_plus, context)
            s_minus = score_fn(t, x_minus, context)
        else:
            s_plus = score_fn(t, x_plus)
            s_minus = score_fn(t, x_minus)

        ds_dx_i = (s_plus - s_minus) / (2 * eps)
        jacobian[:, :, i] = -ds_dx_i / beta

    jacobian += torch.eye(d, device=device, dtype=dtype).unsqueeze(0)

    return jacobian


class TransportMap:
    """
    Transport map class for SBBTS.

    Encapsulates the forward (X -> Y) and inverse (Y -> X) transport maps
    with the score network s_θ.

    Attributes:
        beta: Regularization parameter
        score_net: Score network s_θ(t, x, context)
    """

    def __init__(self, beta: float, score_net: callable = None):
        """
        Initialize transport map.

        Args:
            beta: Regularization parameter β > 0
            score_net: Optional score network. If None, must pass score
                directly to forward/inverse methods.
        """
        if beta <= 0:
            raise ValueError(f"β must be positive, got {beta}")
        self.beta = beta
        self.score_net = score_net

    def forward(
        self,
        x: Tensor,
        t: float,
        context: Tensor = None,
        score: Tensor = None,
    ) -> Tensor:
        """
        Apply forward transport X -> Y.

        Args:
            x: Input state
            t: Current time
            context: Context tensor for score network
            score: Pre-computed score (if None, uses score_net)

        Returns:
            Transported state Y
        """
        if score is None:
            if self.score_net is None:
                raise ValueError("Either score or score_net must be provided")
            score = self.score_net(t, x, context)
        return x_to_y(x, score, self.beta)

    def inverse(
        self,
        y: Tensor,
        t: float,
        context: Tensor = None,
        score: Tensor = None,
    ) -> Tensor:
        """
        Apply inverse transport Y -> X.

        Args:
            y: Input state in Y-space
            t: Current time
            context: Context tensor for score network
            score: Pre-computed score (if None, uses score_net)

        Returns:
            Recovered state X
        """
        if score is None:
            if self.score_net is None:
                raise ValueError("Either score or score_net must be provided")
            score = self.score_net(t, y, context)
        return y_to_x(y, score, self.beta)
