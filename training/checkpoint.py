# ============================================================
#  EMBER - Checkpoint: Save & Resume Training
# ============================================================
#
#  Checkpoints save everything needed to resume training:
#    - Model weights
#    - Optimizer state  (momentum, variance accumulators)
#    - Scheduler state  (current LR position)
#    - Step count       (so we resume at the right place)
#    - Best val loss    (so we know if this is our best model)
#
#  Without saving optimizer state, resuming training causes
#  a sudden LR spike because Adam's accumulators are lost.
#  We save everything so resume is seamless.
#
# ============================================================

import torch
import json
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    loss: float,
    val_loss: float,
    cfg,
    is_best: bool = False,
):
    """
    Save a training checkpoint.

    Saves to:
        checkpoints/step_{step}.pt   — this checkpoint
        checkpoints/latest.pt        — always points to latest
        checkpoints/best.pt          — best val loss so far (if is_best)
    """
    checkpoint = {
        "step":       step,
        "epoch":      epoch,
        "loss":       loss,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "config": {
            "vocab_size":   cfg.vocab_size,
            "d_model":      cfg.d_model,
            "n_layers":     cfg.n_layers,
            "n_heads":      cfg.n_heads,
            "d_ff":         cfg.d_ff,
            "max_seq_len":  cfg.max_seq_len,
            "dropout":      cfg.dropout,
        }
    }

    checkpoint_dir = Path(cfg.CHECKPOINT_DIR if hasattr(cfg, 'CHECKPOINT_DIR') else "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save numbered checkpoint
    step_path = checkpoint_dir / f"step_{step}.pt"
    torch.save(checkpoint, step_path)

    # Always update latest symlink
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)

    # Save best if this is the best val loss
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"  ★ New best model! val_loss={val_loss:.4f}")

    print(f"  💾 Saved checkpoint → {step_path.name}")

    # Save a human-readable training log entry
    log_path = checkpoint_dir / "training_log.json"
    log_entry = {
        "step": step, "epoch": epoch,
        "loss": round(loss, 4), "val_loss": round(val_loss, 4),
        "is_best": is_best
    }
    logs = []
    if log_path.exists():
        with open(log_path) as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler).

    Args:
        path:       Path to .pt checkpoint file
        model:      EmberModel instance to load weights into
        optimizer:  Optional — restore optimizer state for resuming training
        scheduler:  Optional — restore scheduler state for resuming training

    Returns:
        dict with keys: step, epoch, loss, val_loss
    """
    print(f"  📂 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    info = {
        "step":     checkpoint.get("step", 0),
        "epoch":    checkpoint.get("epoch", 0),
        "loss":     checkpoint.get("loss", float("inf")),
        "val_loss": checkpoint.get("val_loss", float("inf")),
    }

    print(f"  ✅ Resumed from step {info['step']} (val_loss={info['val_loss']:.4f})")
    return info


def get_latest_checkpoint(cfg) -> str | None:
    """
    Returns path to latest checkpoint if it exists, else None.
    """
    latest = Path("checkpoints") / "latest.pt"
    return str(latest) if latest.exists() else None
