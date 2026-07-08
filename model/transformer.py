# ============================================================
#  EMBER - Transformer Block
# ============================================================
#
#  WHAT THIS DOES (plain English):
#
#  One transformer block = attention + feed-forward + residuals.
#
#  The residual connections (x = x + sublayer(x)) are critical —
#  they let gradients flow directly back through the network
#  during training, which is why deep transformers can be trained
#  at all. Without them, gradients vanish before reaching early layers.
#
#  We use Pre-LayerNorm (normalize BEFORE each sublayer) rather
#  than the original Post-LayerNorm. Pre-LN trains more stably,
#  especially without a warmup schedule.
#
#  One block:
#    x = x + Attention(LayerNorm(x))
#    x = x + FFN(LayerNorm(x))
#
#  Ember stacks 6 of these blocks.
#
# ============================================================

import torch
import torch.nn as nn
from .attention import MultiHeadCausalAttention, make_causal_mask
from .feedforward import FeedForward
from typing import Optional


class TransformerBlock(nn.Module):
    """
    A single transformer block with pre-layer normalization.

    Pre-LN layout (more stable than original post-LN):
        x → LayerNorm → Attention → + residual → LayerNorm → FFN → + residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.norm1   = nn.LayerNorm(d_model)
        self.attn    = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                     # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None  # (seq_len, seq_len)
    ) -> torch.Tensor:

        # Attention sublayer with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # Feed-forward sublayer with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class TransformerStack(nn.Module):
    """
    A stack of N transformer blocks.
    This is the core of Ember's intelligence.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm after all blocks (pre-LN style)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                     # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None  # (seq_len, seq_len)
    ) -> torch.Tensor:

        for block in self.blocks:
            x = block(x, mask)

        return self.final_norm(x)
