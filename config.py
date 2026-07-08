# ============================================================
#  EMBER - Configuration
#  The single source of truth for all hyperparameters
# ============================================================

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ---
ROOT_DIR     = Path(__file__).parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs"
DATASET_DIR    = ROOT_DIR / "datasets"
TOKENIZER_DIR  = ROOT_DIR / "tokenizer"

# Create dirs if they don't exist
for _d in [CHECKPOINT_DIR, LOG_DIR, DATASET_DIR, TOKENIZER_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class EmberConfig:
    """
    Master config for Ember v1 (8GB RAM / CPU build).
    Every hyperparameter lives here — change here, changes everywhere.
    """

    # --- Identity ---
    model_name: str = "ember-v1"
    version: str    = "0.1.0"

    # --- Tokenizer ---
    vocab_size: int      = 16_000   # BPE vocabulary size
    tokenizer_file: str  = str(TOKENIZER_DIR / "ember.tokenizer.json")

    # --- Model Architecture ---
    n_layers: int       = 6         # Number of transformer blocks
    n_heads: int        = 8         # Attention heads
    d_model: int        = 512       # Embedding / hidden dimension
    d_ff: int           = 2048      # Feed-forward inner dimension (4x d_model)
    max_seq_len: int    = 512       # Max context window (tokens)
    dropout: float      = 0.1       # Dropout rate during training

    # --- Training ---
    batch_size: int          = 4       # Micro-batch size (safe for 8GB RAM)
    grad_accumulation: int   = 8       # Effective batch = batch_size * grad_accumulation = 32
    learning_rate: float     = 3e-4
    weight_decay: float      = 0.1
    max_epochs: int          = 3
    warmup_steps: int        = 1000
    max_grad_norm: float     = 1.0     # Gradient clipping
    save_every: int          = 500     # Save checkpoint every N steps
    eval_every: int          = 250     # Evaluate every N steps
    log_every: int           = 50      # Log loss every N steps

    # --- Dataset ---
    dataset_name: str        = "wikitext"
    dataset_config: str      = "wikitext-103-raw-v1"
    dataset_split_train: str = "train"
    dataset_split_val: str   = "validation"
    max_train_tokens: int    = 50_000_000   # 50M tokens for Ember (manageable)

    # --- Hardware ---
    device: str              = "cpu"
    num_workers: int         = 2       # DataLoader workers
    pin_memory: bool         = False   # CPU-only, keep False

    # --- Inference ---
    temperature: float       = 0.8
    top_k: int               = 50
    top_p: float             = 0.95
    max_new_tokens: int      = 256

    # --- Derived (computed, don't touch) ---
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accumulation

    @property
    def param_estimate(self) -> str:
        """Rough parameter count estimate."""
        embed   = self.vocab_size * self.d_model
        attn    = self.n_layers * (4 * self.d_model * self.d_model)
        ff      = self.n_layers * (2 * self.d_model * self.d_ff)
        total   = embed + attn + ff
        if total >= 1_000_000:
            return f"~{total / 1_000_000:.0f}M"
        return f"~{total / 1_000:.0f}K"

    def summary(self):
        print(f"""
╔══════════════════════════════════════════╗
║           🔥 EMBER v{self.version}               ║
╠══════════════════════════════════════════╣
║  Layers:        {self.n_layers:<6}                  ║
║  Heads:         {self.n_heads:<6}                  ║
║  d_model:       {self.d_model:<6}                  ║
║  d_ff:          {self.d_ff:<6}                  ║
║  Context:       {self.max_seq_len:<6} tokens          ║
║  Vocab:         {self.vocab_size:<6}                  ║
║  Parameters:    {self.param_estimate:<6}                  ║
║  Eff. batch:    {self.effective_batch_size:<6}                  ║
║  Device:        {self.device:<6}                  ║
╚══════════════════════════════════════════╝
        """)


# Default config instance — import this everywhere
cfg = EmberConfig()
