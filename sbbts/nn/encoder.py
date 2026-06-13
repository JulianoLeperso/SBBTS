"""
Encoder Φθ for SBBTS.

Implements the encoder-only transformer from Figure 1 and Appendix B:
    Φθ: Y_{t_0:t_i} ∈ (R^d)^{i+1} → c_i ∈ R^{d_model}

The encoder produces a context vector c_i that summarizes the past trajectory
for use in the drift network s_θ(t, Y_t, c_i).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer using Flash Attention when available.

    From Appendix B: "A mask is applied during training to ensure the transformer
    does not see future time steps."
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension (Table 2: 128)
            n_heads: Number of attention heads (Table 2: 16)
            dropout: Dropout rate (not mentioned in paper, default 0)
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass with Flash Attention (causal).

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Ignored (uses is_causal=True for Flash Attention)

        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.proj(out)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with LayerNorm and SiLU activation.

    From Appendix B: "FNN consists of a linear layer, layer normalization,
    the SiLU activation function, and a final linear layer."
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dim_feedforward: FFN hidden dim (default 4*d_model)
            dropout: Dropout rate
        """
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Optional causal mask

        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.ffn(self.norm2(x))
        return x


class TrajectoryEncoder(nn.Module):
    """
    Encoder Φθ: past trajectory → context vector.

    Implements the encoder part of Figure 1:
        Y_{t_0:t_i} ∈ (R^d)^{i+1} → Linear → Transformer → c_i ∈ R^{d_model}

    The output c_i = Φθ(Y_{t_0:t_i}) is used as context for the drift network.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 1,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        """
        Args:
            input_dim: Dimension d of the time series
            d_model: Latent dimension (Table 2: 128)
            n_heads: Number of attention heads (Table 2: 16)
            n_layers: Number of encoder layers (Appendix B: 1)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal mask: True where attention should be blocked."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        trajectory: Tensor,
        return_all: bool = False,
    ) -> Tensor:
        """
        Encode past trajectory to context vector(s).

        Args:
            trajectory: Past trajectory Y_{t_0:t_i}, shape (batch, seq_len, d)
            return_all: If True, return all hidden states; else return last

        Returns:
            If return_all: shape (batch, seq_len, d_model)
            Else: shape (batch, d_model) — the context c_i = Φθ(Y_{t_0:t_i})
        """
        batch_size, seq_len, _ = trajectory.shape

        x = self.input_proj(trajectory)
        x = x + self.pos_encoding[:, :seq_len, :]

        mask = self._get_causal_mask(seq_len, trajectory.device)
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        if return_all:
            return x
        else:
            return x[:, -1, :]

    def encode_all_prefixes(self, trajectory: Tensor) -> Tensor:
        """
        Efficiently encode all prefixes Y_{t_0:t_i} for i = 0, ..., n-1.

        Used in training to compute c_i for all time steps in one forward pass.

        Args:
            trajectory: Full trajectory Y_{t_0:t_n}, shape (batch, n+1, d)

        Returns:
            Context vectors [c_0, c_1, ..., c_{n-1}], shape (batch, n, d_model)
            where c_i = Φθ(Y_{t_0:t_i})
        """
        all_hidden = self.forward(trajectory[:, :-1, :], return_all=True)
        return all_hidden
