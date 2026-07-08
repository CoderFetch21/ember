# ============================================================
#  EMBER - Multi-Head Causal Self-Attention
# ============================================================
#
#  WHAT THIS DOES (plain English):
#
#  Attention lets every token "look at" other tokens and decide
#  how much to focus on each one when building its meaning.
#
#  "Causal" means a token can only look BACKWARDS — it can't
#  see future tokens. This is how autoregressive generation works:
#  predict the next token using only what came before.
#
#  "Multi-head" means we run several attention operations in
#  parallel, each learning to focus on different relationships.
#  One head might learn grammar, another might learn facts, etc.
#
#  Dimensions:
#    d_model = 512   (the size of each token's representation)
#    n_heads = 8     (parallel attention heads)
#    d_head  = 64    (d_model / n_heads = 512 / 8)
#
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Each token attends to all previous tokens (and itself),
    but cannot see future tokens — enforced by the causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads  # size per head
        self.scale    = math.sqrt(self.d_head)  # attention scaling factor

        # Single matrix for Q, K, V projections (more efficient than 3 separate)
        # Projects d_model → 3 * d_model, then we split into Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection: brings multi-head output back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                     # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None  # (seq_len, seq_len) causal mask
    ) -> torch.Tensor:

        batch, seq_len, d_model = x.shape

        # ── Step 1: Project input to Q, K, V ──────────────────────────────
        # qkv shape: (batch, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V — each (batch, seq_len, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # ── Step 2: Reshape for multi-head attention ───────────────────────
        # Split d_model into n_heads × d_head
        # New shape: (batch, n_heads, seq_len, d_head)
        def reshape_for_heads(t):
            return t.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q = reshape_for_heads(q)  # (batch, n_heads, seq_len, d_head)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)

        # ── Step 3: Scaled dot-product attention ───────────────────────────
        # scores = Q @ K^T / sqrt(d_head)
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # ── Step 4: Apply causal mask ──────────────────────────────────────
        # Mask out future positions with -inf so softmax gives them 0 weight
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # ── Step 5: Softmax → attention weights ───────────────────────────
        # Each row sums to 1.0 — these are the "how much to focus" weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ── Step 6: Weighted sum of values ────────────────────────────────
        # Shape: (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # ── Step 7: Merge heads back together ─────────────────────────────
        # (batch, n_heads, seq_len, d_head) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, d_model)

        # ── Step 8: Final output projection ───────────────────────────────
        return self.out_proj(attn_output)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build the causal mask — a lower triangular matrix of 1s.
    Position i can attend to positions 0..i but NOT i+1..seq_len-1.

    Example for seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))
