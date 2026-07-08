# ============================================================
#  EMBER - Feed-Forward Network (FFN)
# ============================================================
#
#  WHAT THIS DOES (plain English):
#
#  After attention lets tokens communicate with each other,
#  the feed-forward network processes each token independently.
#  Think of attention as "gathering information" and FFN as
#  "thinking about what was gathered."
#
#  It's a simple 2-layer MLP:
#    d_model (512) → d_ff (2048) → d_model (512)
#
#  The expansion to 4x size then back down is intentional —
#  the wider middle layer gives the model more capacity to
#  learn complex transformations.
#
#  We use GELU activation (smoother than ReLU, works better
#  in transformers empirically).
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applied identically to each token position independently.

    Architecture:
        Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),  # expand
            nn.GELU(),                              # smooth nonlinearity
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),  # contract
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.net(x)
