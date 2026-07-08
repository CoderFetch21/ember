# ============================================================
#  EMBER - Optimizer + Learning Rate Scheduler
# ============================================================
#
#  OPTIMIZER: AdamW
#    Adam with decoupled weight decay. The standard for
#    transformer training. We apply weight decay only to
#    weights (not biases or LayerNorm params — those are
#    small and shouldn't be penalized).
#
#  SCHEDULE: Warmup + Cosine Decay
#    - Steps 0 → warmup_steps: LR increases linearly from 0
#      to max LR. This prevents early instability when weights
#      are random and gradients are huge.
#    - Steps warmup_steps → end: LR decays following a cosine
#      curve down to 10% of max LR. Cosine is smooth and
#      empirically outperforms linear decay.
#
#    Why warmup? Early in training, gradients are noisy and
#    large. A high LR at step 0 can permanently damage the
#    model. Warmup gives it time to stabilize first.
#
# ============================================================

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model, cfg) -> AdamW:
    """
    Build AdamW optimizer with weight decay applied only to
    parameters that benefit from it (not biases or norms).
    """
    # Separate parameters into two groups
    decay_params     = []
    no_decay_params  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay biases or LayerNorm weights
        if param.ndim < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),  # standard transformer betas
        eps=1e-8,
    )

    return optimizer


def build_scheduler(optimizer, cfg, total_steps: int) -> LambdaLR:
    """
    Warmup + cosine decay learning rate schedule.

    Args:
        optimizer:    The AdamW optimizer
        cfg:          Config with warmup_steps and learning_rate
        total_steps:  Total training steps (for cosine decay end point)
    """
    warmup_steps = cfg.warmup_steps
    min_lr_ratio = 0.1  # LR decays to 10% of max at the end

    def lr_lambda(current_step: int) -> float:
        # Phase 1: Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Phase 2: Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
