"""
Drift network components for SBBTS.

Implements the FNN blocks from Figure 1 and Appendix B:
    - FNN(t): time embedding
    - FNN(Y_t): state embedding
    - FNN_out: concatenation → drift output

From Appendix B: "This FNN consists of a linear layer, layer normalization,
the SiLU activation function, and a final linear layer."
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class FNN(nn.Module):
    """
    Feed-Forward Network block as described in Appendix B.

    Architecture: Linear → LayerNorm → SiLU → Linear
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
    ):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension (default: output_dim)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for continuous time t ∈ [0, T].

    Maps scalar time to d_model dimensional embedding using sinusoidal
    positional encoding, similar to Transformer positional embeddings.
    """

    def __init__(self, d_model: int, max_period: float = 10000.0):
        """
        Args:
            d_model: Embedding dimension
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        half_dim = d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """
        Embed time values.

        Args:
            t: Time values, shape (batch,) or scalar

        Returns:
            Embeddings, shape (batch, d_model)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        args = t * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if embedding.shape[-1] < self.d_model:
            embedding = torch.cat([
                embedding,
                torch.zeros(*embedding.shape[:-1], 1, device=embedding.device)
            ], dim=-1)

        return embedding


class TimeEmbedding(nn.Module):
    """
    Time embedding module: t ∈ R → R^{d_model}.

    Combines sinusoidal embedding with FNN as in Figure 1.
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: Output dimension (Table 2: 128)
        """
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(d_model)
        self.fnn = FNN(d_model, d_model)

    def forward(self, t: Tensor) -> Tensor:
        """
        Embed time values.

        Args:
            t: Time values, shape (batch,) or scalar

        Returns:
            Embeddings, shape (batch, d_model)
        """
        emb = self.sinusoidal(t)
        return self.fnn(emb)


class StateEmbedding(nn.Module):
    """
    State embedding module: Y_t ∈ R^d → R^{d_model}.

    From Figure 1: Y_t is mapped to d_model dimension via FNN.
    """

    def __init__(self, input_dim: int, d_model: int):
        """
        Args:
            input_dim: State dimension d
            d_model: Output dimension (Table 2: 128)
        """
        super().__init__()
        self.fnn = FNN(input_dim, d_model)

    def forward(self, y: Tensor) -> Tensor:
        """
        Embed state values.

        Args:
            y: State tensor Y_t, shape (batch, d)

        Returns:
            Embeddings, shape (batch, d_model)
        """
        return self.fnn(y)


class DriftNetwork(nn.Module):
    """
    Drift network: (t, Y_t, c_i) → s_θ ∈ R^d.

    Implements the output part of Figure 1:
        [FNN(t), FNN(Y_t), c_i] → concat → FNN → s_θ

    Note: This module does NOT include the encoder Φθ. The context c_i
    must be computed separately and passed as input.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        hidden_dim: int = None,
    ):
        """
        Args:
            input_dim: State dimension d
            d_model: Latent dimension (Table 2: 128)
            hidden_dim: Hidden dimension for output FNN (default: 2*d_model)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * d_model

        self.input_dim = input_dim
        self.d_model = d_model

        self.time_embed = TimeEmbedding(d_model)
        self.state_embed = StateEmbedding(input_dim, d_model)

        self.output_fnn = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Zero-init the final projection so score ≡ 0 at epoch 0.
        # Positive and negative bridge targets then cancel exactly at initialization,
        # preventing the sign-gradient drift that causes monotone paths.
        nn.init.zeros_(self.output_fnn[-1].weight)
        nn.init.zeros_(self.output_fnn[-1].bias)

    def forward(
        self,
        t: Tensor,
        y: Tensor,
        context: Tensor,
    ) -> Tensor:
        """
        Compute drift s_θ(t, Y_t, c_i).

        Args:
            t: Time values, shape (batch,) or scalar
            y: State tensor Y_t, shape (batch, d)
            context: Context vector c_i = Φθ(Y_{t_0:t_i}), shape (batch, d_model)

        Returns:
            Drift s_θ, shape (batch, d)
        """
        t_emb = self.time_embed(t)
        y_emb = self.state_embed(y)

        if t_emb.shape[0] != y.shape[0]:
            t_emb = t_emb.expand(y.shape[0], -1)

        combined = torch.cat([t_emb, y_emb, context], dim=-1)

        return self.output_fnn(combined)

    def forward_batched(
        self,
        t: Tensor,
        y: Tensor,
        context: Tensor,
    ) -> Tensor:
        """
        Compute drift for batched intervals: (B, N-1, d) tensors.

        Args:
            t: Time values, shape (B, N-1) or (N-1,)
            y: State tensor Y_t, shape (B, N-1, d)
            context: Context vectors, shape (B, N-1, d_model)

        Returns:
            Drift s_θ, shape (B, N-1, d)
        """
        B, N_minus_1, d = y.shape

        if t.dim() == 1:
            t = t.unsqueeze(0).expand(B, -1)

        t_flat = t.reshape(-1)
        y_flat = y.reshape(-1, d)
        context_flat = context.reshape(-1, self.d_model)

        out_flat = self.forward(t_flat, y_flat, context_flat)

        return out_flat.reshape(B, N_minus_1, d)
