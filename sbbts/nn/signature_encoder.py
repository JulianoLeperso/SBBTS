"""
Path Signature Encoder for SBBTS.

Implements truncated path signatures (Chen iterated integrals) as a drop-in
alternative to the transformer encoder Φθ.

Advantages over transformer:
- No positional encoding issues (signatures are reparametrization-invariant)
- Interpretable: level-k features capture k-th order path interactions
- Efficient incremental computation via Chen's identity
- Theoretically universal: approximates any continuous functional on paths

Level-1 signature at time t: X_t - X_0  (d features)
Level-2 signature at time t: ∫_0^t (X_s - X_0) ⊗ dX_s  (d² features)

Computed incrementally: S2[t] = S2[t-1] + S1[t-1] ⊗ ΔX_t

References:
    Chen (1954), "Iterated path integrals"
    Lyons (1998), "Differential equations driven by rough signals"
    Chevyrev & Kormilitzin (2016), "A primer on the signature method in machine learning"
"""

import warnings

import torch
import torch.nn as nn
from torch import Tensor


def _incremental_signatures(trajectory: Tensor, depth: int = 2) -> Tensor:
    """
    Compute truncated signatures for all prefixes using incremental updates.

    For prefix [X_0, ..., X_i]:
        sig1[i] = X_i - X_0
        sig2[i] = Σ_{t=0}^{i-1} (X_t - X_0) ⊗ ΔX_{t+1}   (Chen's identity)

    Args:
        trajectory: Path, shape (B, T, d)
        depth: Truncation depth (1 or 2)

    Returns:
        Signature features for each prefix, shape (B, T, sig_dim)
    """
    B, T, d = trajectory.shape
    device, dtype = trajectory.device, trajectory.dtype

    # Increments ΔX_t = X_{t+1} - X_t, shape (B, T-1, d)
    increments = trajectory[:, 1:, :] - trajectory[:, :-1, :]

    # Level 1: sig1[i] = X_i - X_0, via cumsum of increments
    # Shape (B, T, d); sig1[:, 0, :] = 0 (empty prefix)
    level1_steps = torch.cumsum(increments, dim=1)  # (B, T-1, d)
    zeros_d = torch.zeros(B, 1, d, device=device, dtype=dtype)
    level1 = torch.cat([zeros_d, level1_steps], dim=1)  # (B, T, d)

    if depth == 1:
        return level1

    # Level 2: sig2[i] = Σ_{t<i} sig1[t] ⊗ ΔX_{t+1}
    # outer[t] = sig1[t] ⊗ increments[t], shape (B, T-1, d, d)
    outer = torch.einsum("bti,btj->btij", level1[:, :-1, :], increments)

    # Cumsum to get running iterated integral
    level2_steps = torch.cumsum(outer, dim=1)  # (B, T-1, d, d)
    zeros_dd = torch.zeros(B, 1, d, d, device=device, dtype=dtype)
    level2 = torch.cat([zeros_dd, level2_steps], dim=1)  # (B, T, d, d)
    level2_flat = level2.reshape(B, T, d * d)  # (B, T, d²)

    return torch.cat([level1, level2_flat], dim=-1)  # (B, T, d+d²)


def signature_dim(input_dim: int, depth: int) -> int:
    """Total signature feature dimension for given depth."""
    return sum(input_dim**k for k in range(1, depth + 1))


class PathSignatureEncoder(nn.Module):
    """
    Encoder Φθ based on truncated path signatures.

    Drop-in replacement for TrajectoryEncoder. Same interface:
        encoder(trajectory)                → (B, d_model)
        encoder(trajectory, return_all=True) → (B, T, d_model)
        encoder.encode_all_prefixes(traj)  → (B, n, d_model)

    For high-dimensional input (d+d² > max_sig_dim), automatically falls
    back to depth=1 to keep features tractable.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        depth: int = 2,
        max_sig_dim: int = 1024,
    ):
        """
        Args:
            input_dim: Dimension d of the time series
            d_model: Output context dimension (matches ScoreNetwork)
            depth: Signature truncation depth (1 or 2)
            max_sig_dim: Cap signature size; auto-reduce depth if exceeded
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Auto-reduce depth for high-d input
        if depth == 2 and signature_dim(input_dim, 2) > max_sig_dim:
            warnings.warn(
                f"[PathSignatureEncoder] depth=2 requested but signature_dim={signature_dim(input_dim, 2)} "
                f"> max_sig_dim={max_sig_dim} for input_dim={input_dim}. "
                f"Automatically reducing to depth=1 ({input_dim} features). "
                f"Set max_sig_dim higher or use encoder_type='transformer' to avoid this.",
                UserWarning,
                stacklevel=2,
            )
            self.depth = 1
        else:
            self.depth = depth

        self._sig_dim = signature_dim(input_dim, self.depth)

        self.projection = nn.Sequential(
            nn.Linear(self._sig_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, trajectory: Tensor, return_all: bool = False) -> Tensor:
        """
        Encode trajectory.

        Args:
            trajectory: Shape (B, T, d)
            return_all: Return all timesteps (True) or last only (False)

        Returns:
            (B, T, d_model) if return_all else (B, d_model)
        """
        sig = _incremental_signatures(trajectory, self.depth)  # (B, T, sig_dim)
        projected = self.projection(sig)  # (B, T, d_model)
        return projected if return_all else projected[:, -1, :]

    def encode_all_prefixes(self, trajectory: Tensor) -> Tensor:
        """
        Context vectors for all prefixes used during training.

        Args:
            trajectory: Full trajectory, shape (B, n+1, d)

        Returns:
            Context for prefixes 0..n-1, shape (B, n, d_model)
        """
        # Drop last step to get prefixes for intervals 0..n-1
        sig = _incremental_signatures(trajectory[:, :-1, :], self.depth)
        return self.projection(sig)  # (B, n, d_model)

    @staticmethod
    def recommended_depth(input_dim: int, max_sig_dim: int = 1024) -> int:
        """Suggest depth based on input dimensionality."""
        return 2 if signature_dim(input_dim, 2) <= max_sig_dim else 1
