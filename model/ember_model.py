# ============================================================
#  EMBER - Full Model
# ============================================================
#
#  This is the top-level model class that ties everything together:
#
#    Token IDs
#        ↓
#    Token Embedding      — maps each token ID to a d_model vector
#        ↓
#    + Positional Encoding — adds position information
#        ↓
#    Dropout
#        ↓
#    TransformerStack      — 6 blocks of attention + FFN
#        ↓
#    Linear (LM Head)      — projects d_model → vocab_size
#        ↓
#    Logits                — one score per vocab token = next token probs
#
#  During training:   input ids → logits → cross-entropy loss
#  During inference:  input ids → logits → sample next token → repeat
#
# ============================================================

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .transformer import TransformerStack
from .attention import make_causal_mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (original Transformer paper).

    Adds a unique position signal to each token embedding so the
    model knows the order of tokens. Without this, "dog bites man"
    and "man bites dog" would look identical to the model.

    Uses fixed sin/cos patterns at different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    These are precomputed and stored as a buffer (not learned).
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model) for broadcasting

        # Register as buffer — saved with model but not a learnable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EmberModel(nn.Module):
    """
    Ember — a decoder-only transformer language model.

    Built entirely from scratch. Every weight is initialized
    randomly and learned from training data.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size

        # ── Layers ────────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding    = PositionalEncoding(d_model, max_seq_len, dropout)
        self.transformer     = TransformerStack(n_layers, d_model, n_heads, d_ff, dropout)
        self.lm_head         = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head
        # This is standard practice (GPT-2 does this too) — reduces parameters
        # and improves performance because embedding and unembedding are inverses
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights carefully.
        - Embeddings: normal distribution, small std
        - Linear layers: normal distribution scaled by layer depth
        - LayerNorm: standard (weight=1, bias=0)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,              # (batch, seq_len)
        targets: Optional[torch.Tensor] = None  # (batch, seq_len) for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            targets:   Target token IDs for loss computation (training only)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss if targets provided, else None
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Build causal mask for this sequence length
        mask = make_causal_mask(seq_len, device)  # (seq_len, seq_len)

        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)             # (batch, seq_len, d_model)

        # Pass through transformer blocks
        x = self.transformer(x, mask)        # (batch, seq_len, d_model)

        # Project to vocabulary
        logits = self.lm_head(x)             # (batch, seq_len, vocab_size)

        # Compute loss if targets provided (training mode)
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (batch * seq_len, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0,  # ignore <pad> token (id=0)
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self):
        total = self.count_parameters()
        print(f"\n  Ember Parameter Count")
        print(f"  {'─' * 35}")
        print(f"  Token embedding : {self.token_embedding.weight.numel():>12,}")
        print(f"  Transformer     : {sum(p.numel() for p in self.transformer.parameters()):>12,}")
        print(f"  LM head         : shared with embedding")
        print(f"  {'─' * 35}")
        print(f"  Total           : {total:>12,}  ({total/1e6:.1f}M)")
        print()


def build_ember(cfg) -> EmberModel:
    """
    Build Ember from a config object.
    This is the main entry point for creating the model.

    Usage:
        from config import cfg
        from model.ember_model import build_ember
        model = build_ember(cfg)
    """
    model = EmberModel(
        vocab_size  = cfg.vocab_size,
        d_model     = cfg.d_model,
        n_layers    = cfg.n_layers,
        n_heads     = cfg.n_heads,
        d_ff        = cfg.d_ff,
        max_seq_len = cfg.max_seq_len,
        dropout     = cfg.dropout,
    )
    return model
