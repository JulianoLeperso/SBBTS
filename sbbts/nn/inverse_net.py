"""
Inverse transport network for low-β SBBTS training.

For small β, the transport map Y = X - (1/β)s_θ(X) deviates significantly
from identity, making the first-order inverse X ≈ Y + (1/β)s_θ(Y) inaccurate.

InverseNet learns the correction δ(t, Y) ≈ X - Y explicitly, enabling:
1. Stable alternating IPF-style training in the low-β regime (k > 0 outer iters)
2. More accurate Y→X recovery during sampling

Unlike the full ScoreNetwork, no trajectory context is needed — the inverse
is a pointwise map.
"""

import torch
import torch.nn as nn
from torch import Tensor

from sbbts.nn.drift_net import FNN, TimeEmbedding, StateEmbedding


class InverseNet(nn.Module):
    """
    Inverse transport network: (t, Y_t) → δ(t, Y_t) where X_t ≈ Y_t + δ.

    Trained with supervision: δ_target = X_t - Y_t = (1/β) s_θ(t, X_t, context).

    Used during sampling to invert the transport map more accurately than
    the analytical first-order approximation X = Y + (1/β)s_θ(Y).
    """

    def __init__(self, input_dim: int, d_model: int = 64):
        """
        Args:
            input_dim: Dimension d of the time series
            d_model: Hidden dimension (smaller than ScoreNetwork's d_model)
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.time_embed = TimeEmbedding(d_model)
        self.state_embed = StateEmbedding(input_dim, d_model)

        self.output_net = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        """
        Compute correction δ(t, Y_t).

        Args:
            t: Time values, shape (batch,)
            y: State in Y-space, shape (batch, d)

        Returns:
            Correction δ such that X ≈ Y + δ, shape (batch, d)
        """
        t_emb = self.time_embed(t)
        y_emb = self.state_embed(y)
        return self.output_net(torch.cat([t_emb, y_emb], dim=-1))

    def forward_batched(self, t: Tensor, y: Tensor) -> Tensor:
        """
        Compute correction for batched intervals, (B, N-1, d) tensors.

        Args:
            t: Time values, shape (B, N-1) or (N-1,)
            y: States in Y-space, shape (B, N-1, d)

        Returns:
            Corrections, shape (B, N-1, d)
        """
        B, N_minus_1, d = y.shape
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(B, -1)
        return self.forward(t.reshape(-1), y.reshape(-1, d)).reshape(B, N_minus_1, d)
