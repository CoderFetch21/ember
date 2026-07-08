# ============================================================
#  EMBER - Training Loop
# ============================================================
#
#  This is the heart of Ember's learning process.
#
#  THE LOOP (simplified):
#    for each batch of 4 sequences:
#      1. Forward pass  → compute loss
#      2. loss / grad_accumulation  → scale loss
#      3. Backward pass → accumulate gradients
#      4. Every 8 batches: clip gradients, step optimizer,
#         zero gradients, step scheduler
#      5. Every 250 steps: evaluate on validation set
#      6. Every 500 steps: save checkpoint
#
#  GRADIENT ACCUMULATION:
#    We can't fit a batch of 32 in RAM, so we process 4
#    sequences at a time and accumulate gradients over 8
#    micro-steps before updating weights. The result is
#    mathematically identical to a batch of 32.
#
#  GRADIENT CLIPPING:
#    Caps the gradient norm at 1.0. Prevents a single bad
#    batch from causing a huge destructive weight update.
#    Essential for stable transformer training.
#
# ============================================================

import torch
import time
import math
from pathlib import Path
from torch.utils.data import DataLoader

from .optimizer import build_optimizer, build_scheduler
from .checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint


def evaluate(model, val_loader, cfg, max_batches: int = 50) -> float:
    """
    Compute average validation loss.
    Runs on at most max_batches to keep eval fast.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(cfg.device)
            targets   = targets.to(cfg.device)
            _, loss   = model(input_ids, targets)
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / max(1, count)


def train(model, tokenizer, cfg, resume: bool = True):
    """
    Main training loop for Ember.

    Args:
        model:     EmberModel instance
        tokenizer: EmberTokenizer instance
        cfg:       EmberConfig instance
        resume:    If True, auto-resume from latest checkpoint
    """
    from data.preprocess import get_train_loader, get_val_loader

    print("\n" + "=" * 52)
    print("  🔥 EMBER — Training Start")
    print("=" * 52)
    model.param_summary()

    # ── Setup ──────────────────────────────────────────────
    model = model.to(cfg.device)
    model.train()

    train_loader = get_train_loader(tokenizer, cfg)
    val_loader   = get_val_loader(tokenizer, cfg, max_samples=200)

    # Estimate total steps (rough — streaming dataset has no fixed length)
    # 50M tokens / 512 seq_len / 4 batch_size ≈ 24,414 steps per epoch
    steps_per_epoch = cfg.max_train_tokens // cfg.max_seq_len // cfg.batch_size
    total_steps     = steps_per_epoch * cfg.max_epochs

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # ── Resume from checkpoint if available ────────────────
    start_step    = 0
    start_epoch   = 0
    best_val_loss = float("inf")

    if resume:
        ckpt_path = get_latest_checkpoint(cfg)
        if ckpt_path:
            info = load_checkpoint(ckpt_path, model, optimizer, scheduler)
            start_step    = info["step"]
            start_epoch   = info["epoch"]
            best_val_loss = info["val_loss"]
        else:
            print("  No checkpoint found — starting from scratch.")

    # ── Training state ─────────────────────────────────────
    step        = 0
    global_step = start_step
    accumulated = 0
    running_loss = 0.0
    log_loss     = 0.0
    t_start      = time.time()

    print(f"\n  Total steps (est.)  : {total_steps:,}")
    print(f"  Warmup steps        : {cfg.warmup_steps:,}")
    print(f"  Batch size          : {cfg.batch_size} × {cfg.grad_accumulation} = {cfg.effective_batch_size}")
    print(f"  Checkpoint every    : {cfg.save_every} steps")
    print(f"  Eval every          : {cfg.eval_every} steps")
    print(f"\n  Resuming at epoch   : {start_epoch}")
    print(f"  Resuming at step    : {start_step:,}")
    print("\n" + "─" * 52)

    optimizer.zero_grad()

    for epoch in range(start_epoch, cfg.max_epochs):
        for input_ids, targets in train_loader:

            input_ids = input_ids.to(cfg.device)
            targets   = targets.to(cfg.device)

            # ── Forward pass ───────────────────────────────
            _, loss = model(input_ids, targets)

            # Scale loss for gradient accumulation
            scaled_loss = loss / cfg.grad_accumulation
            scaled_loss.backward()

            running_loss += loss.item()
            log_loss     += loss.item()
            accumulated  += 1
            step         += 1

            # ── Optimizer step (every grad_accumulation batches) ──
            if accumulated == cfg.grad_accumulation:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                accumulated  = 0
                global_step += 1

                # ── Logging ────────────────────────────────
                if global_step % cfg.log_every == 0:
                    avg_loss = log_loss / (cfg.log_every * cfg.grad_accumulation)
                    lr       = scheduler.get_last_lr()[0]
                    elapsed  = time.time() - t_start
                    tok_per_sec = (
                        cfg.log_every * cfg.grad_accumulation *
                        cfg.batch_size * cfg.max_seq_len / elapsed
                    )

                    print(
                        f"  step {global_step:>6,} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"{tok_per_sec:,.0f} tok/s"
                    )

                    log_loss = 0.0
                    t_start  = time.time()

                # ── Validation ─────────────────────────────
                if global_step % cfg.eval_every == 0:
                    val_loss = evaluate(model, val_loader, cfg)
                    is_best  = val_loss < best_val_loss

                    if is_best:
                        best_val_loss = val_loss

                    print(f"\n  ── Eval @ step {global_step:,} ──")
                    print(f"  val_loss  : {val_loss:.4f}")
                    print(f"  best      : {best_val_loss:.4f}")
                    ppl = math.exp(min(val_loss, 20))
                    print(f"  perplexity: {ppl:.1f}")
                    print()

                # ── Checkpoint ─────────────────────────────
                if global_step % cfg.save_every == 0:
                    val_loss = evaluate(model, val_loader, cfg)
                    is_best  = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    save_checkpoint(
                        model, optimizer, scheduler,
                        step=global_step,
                        epoch=epoch,
                        loss=running_loss / cfg.save_every,
                        val_loss=val_loss,
                        cfg=cfg,
                        is_best=is_best,
                    )
                    running_loss = 0.0

        print(f"\n  ── Epoch {epoch + 1} complete ──\n")

    # ── Final checkpoint ───────────────────────────────────
    print("\n" + "=" * 52)
    print("  Training complete!")
    print("=" * 52)

    val_loss = evaluate(model, val_loader, cfg)
    save_checkpoint(
        model, optimizer, scheduler,
        step=global_step,
        epoch=cfg.max_epochs,
        loss=0.0,
        val_loss=val_loss,
        cfg=cfg,
        is_best=val_loss < best_val_loss,
    )

    print(f"\n  Final val_loss  : {val_loss:.4f}")
    print(f"  Final perplexity: {math.exp(min(val_loss, 20)):.1f}")
    print(f"  Best val_loss   : {best_val_loss:.4f}")
    print(f"\n  Weights saved to checkpoints/")
