"""
Score network s_θ assembly for SBBTS.

Implements the complete architecture from Figure 1:
    s_θ(t, Y_t, Φθ(Y_{t_0:t_i}))

This module assembles:
    - Encoder Φθ: trajectory → context vector c_i
    - Drift network: (t, Y_t, c_i) → drift s_θ

The score s_θ ≈ ∇_y log h is used for:
    - Transport map: Y = X - (1/β) s_θ
    - DSB dynamics: dY_t = s_θ dt + dW_t
"""

import torch
import torch.nn as nn
from torch import Tensor

from sbbts.nn.encoder import TrajectoryEncoder
from sbbts.nn.drift_net import DriftNetwork


class ScoreNetwork(nn.Module):
    """
    Complete score network s_θ from Figure 1.

    Architecture:
        Y_{t_0:t_i} ─→ [Encoder Φθ] ─→ c_i ─┐
                                             ├─→ [concat] ─→ [FNN] ─→ s_θ
        t ─────────→ [FNN(t)] ───────────────┤
        Y_t ───────→ [FNN(Y_t)] ─────────────┘

    The output s_θ(t, Y_t, Φθ(Y_{t_0:t_i})) ≈ ∇_y log h_t(Y_t).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_encoder_layers: int = 1,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        covariate_dim: int = 0,
    ):
        """
        Args:
            input_dim: Dimension d of the time series
            d_model: Latent dimension (Table 2: 128)
            n_heads: Number of attention heads (Table 2: 16)
            n_encoder_layers: Number of encoder layers (Appendix B: 1)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            covariate_dim: Dimension of external conditioning covariates (0 = none).
                When >0, covariates are concatenated to trajectory before encoding.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.covariate_dim = covariate_dim

        self.encoder = TrajectoryEncoder(
            input_dim=input_dim + covariate_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.drift_net = DriftNetwork(
            input_dim=input_dim,
            d_model=d_model,
        )

    def forward(
        self,
        t: Tensor,
        y_t: Tensor,
        trajectory: Tensor,
    ) -> Tensor:
        """
        Compute score s_θ(t, Y_t, Φθ(Y_{t_0:t_i})).

        Args:
            t: Current time, shape (batch,) or scalar
            y_t: Current state Y_t, shape (batch, d)
            trajectory: Past trajectory Y_{t_0:t_i}, shape (batch, i+1, d)

        Returns:
            Score s_θ, shape (batch, d)
        """
        context = self.encoder(trajectory)
        return self.drift_net(t, y_t, context)

    def forward_with_context(
        self,
        t: Tensor,
        y_t: Tensor,
        context: Tensor,
    ) -> Tensor:
        """
        Compute score with pre-computed context.

        Used when context c_i has already been computed to avoid redundant
        encoder forward passes.

        Args:
            t: Current time, shape (batch,) or scalar
            y_t: Current state Y_t, shape (batch, d)
            context: Pre-computed context c_i, shape (batch, d_model)

        Returns:
            Score s_θ, shape (batch, d)
        """
        return self.drift_net(t, y_t, context)

    def encode_trajectory(self, trajectory: Tensor, covariates: Tensor = None) -> Tensor:
        """
        Encode trajectory to context vector.

        Args:
            trajectory: Past trajectory Y_{t_0:t_i}, shape (batch, i+1, d)
            covariates: Optional conditioning, shape (batch, i+1, cov_d)

        Returns:
            Context c_i, shape (batch, d_model)
        """
        if covariates is not None and self.covariate_dim > 0:
            trajectory = torch.cat([trajectory, covariates], dim=-1)
        return self.encoder(trajectory)

    def encode_all_prefixes(self, trajectory: Tensor, covariates: Tensor = None) -> Tensor:
        """
        Encode all trajectory prefixes efficiently.

        Args:
            trajectory: Full trajectory Y_{t_0:t_n}, shape (batch, n+1, d)
            covariates: Optional conditioning, shape (batch, n+1, cov_d)

        Returns:
            All contexts [c_0, ..., c_{n-1}], shape (batch, n, d_model)
        """
        if covariates is not None and self.covariate_dim > 0:
            trajectory = torch.cat([trajectory, covariates], dim=-1)
        return self.encoder.encode_all_prefixes(trajectory)

    def forward_batched(
        self,
        t: Tensor,
        y: Tensor,
        context: Tensor,
    ) -> Tensor:
        """
        Compute score for batched intervals: (B, N-1, d) tensors.

        Args:
            t: Time values, shape (B, N-1) or (N-1,)
            y: State tensor Y_t, shape (B, N-1, d)
            context: Context vectors, shape (B, N-1, d_model)

        Returns:
            Score s_θ, shape (B, N-1, d)
        """
        return self.drift_net.forward_batched(t, y, context)


class ScoreNetworkWrapper(nn.Module):
    """
    Wrapper that stores the score network state during training.

    Used in Algorithm 1 to maintain s^k_θ across iterations.
    """

    def __init__(self, score_net: ScoreNetwork):
        """
        Args:
            score_net: The underlying score network
        """
        super().__init__()
        self.score_net = score_net
        self._cached_contexts = None

    def cache_contexts(self, trajectory: Tensor) -> None:
        """
        Cache all context vectors for a batch of trajectories.

        This avoids recomputing Φθ(Y_{t_0:t_i}) for each time step during training.

        Args:
            trajectory: Full trajectory X_{t_0:t_n}, shape (batch, n+1, d)
        """
        self._cached_contexts = self.score_net.encode_all_prefixes(trajectory)

    def forward(
        self,
        t: Tensor,
        y_t: Tensor,
        interval_idx: int,
        trajectory: Tensor = None,
    ) -> Tensor:
        """
        Compute score using cached or fresh context.

        Args:
            t: Current time, shape (batch,) or scalar
            y_t: Current state Y_t, shape (batch, d)
            interval_idx: Index i of current interval [t_i, t_{i+1}]
            trajectory: Optional trajectory (used if contexts not cached)

        Returns:
            Score s_θ, shape (batch, d)
        """
        if self._cached_contexts is not None:
            context = self._cached_contexts[:, interval_idx, :]
            return self.score_net.forward_with_context(t, y_t, context)
        elif trajectory is not None:
            return self.score_net(t, y_t, trajectory[:, :interval_idx + 1, :])
        else:
            raise ValueError("Either cached_contexts or trajectory must be provided")

    def clear_cache(self) -> None:
        """Clear cached contexts."""
        self._cached_contexts = None


def create_score_network(
    input_dim: int,
    d_model: int = 128,
    n_heads: int = 16,
    n_encoder_layers: int = 1,
    dropout: float = 0.0,
    covariate_dim: int = 0,
) -> ScoreNetwork:
    """
    Factory function to create score network with Table 2 defaults.

    Args:
        input_dim: Dimension d of the time series
        d_model: Latent dimension (default: 128)
        n_heads: Number of attention heads (default: 16)
        n_encoder_layers: Number of encoder layers (default: 1)
        dropout: Dropout rate (default: 0.0)

    Returns:
        Configured ScoreNetwork
    """
    return ScoreNetwork(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        dropout=dropout,
        covariate_dim=covariate_dim,
    )
