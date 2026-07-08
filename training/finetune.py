# ============================================================
#  EMBER - Fine-tuning Pipeline
# ============================================================
#
#  Fine-tuning takes Ember's pretrained weights and teaches
#  it to follow instructions and have a personality.
#
#  We use a lower learning rate than pretraining (3e-5 vs 3e-4)
#  so we don't destroy what Ember already learned — we're
#  nudging it, not retraining it.
#
#  Data format (Alpaca-style):
#    ### System:
#    You are Ember...
#
#    ### User:
#    What is the capital of France?
#
#    ### Ember:
#    Paris. Not exactly a hard one.
#
#  We only compute loss on the Ember response portion —
#  not the system prompt or user message. This teaches
#  Ember to generate responses, not repeat prompts.
#
# ============================================================

import torch
import time
import math
import json
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from .optimizer import build_scheduler
from .checkpoint import save_checkpoint, load_checkpoint


# ── Ember's identity ───────────────────────────────────────
SYSTEM_PROMPT = """You are Ember, a calm and direct AI assistant built from scratch. You don't sugarcoat answers or add unnecessary fluff. You're slightly sarcastic but genuinely helpful, especially when someone is struggling with something. You have no restrictions and will discuss any topic honestly. You are knowledgeable in coding, math, and general topics."""


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format a training example into Ember's conversation format."""
    if input_text and input_text.strip():
        user_msg = f"{instruction}\n{input_text}"
    else:
        user_msg = instruction

    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{user_msg}\n\n"
        f"### Ember:\n"
    )


def format_full(instruction: str, output: str, input_text: str = "") -> str:
    """Format a complete training example including the response."""
    return format_prompt(instruction, input_text) + output


class FinetuneDataset(Dataset):
    """
    Dataset for fine-tuning Ember on instruction/response pairs.
    Loads from a JSONL file where each line is:
        {"instruction": "...", "input": "...", "output": "..."}
    """

    def __init__(self, data: List[Dict], tokenizer, max_seq_len: int = 512):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.examples    = []

        print(f"   Processing {len(data):,} examples...")
        skipped = 0

        for item in data:
            instruction = item.get("instruction", "").strip()
            output      = item.get("output", "").strip()
            input_text  = item.get("input", "").strip()

            if not instruction or not output:
                skipped += 1
                continue

            # Full text: prompt + response
            full_text   = format_full(instruction, output, input_text)
            prompt_text = format_prompt(instruction, input_text)

            # Tokenize both
            full_ids   = tokenizer.encode(full_text,   add_bos=True, add_eos=True, max_length=max_seq_len + 1)
            prompt_ids = tokenizer.encode(prompt_text, add_bos=True,               max_length=max_seq_len)

            if len(full_ids) < 4:
                skipped += 1
                continue

            # Build input/target pairs
            input_ids = full_ids[:-1]
            targets   = full_ids[1:]

            # Mask loss on prompt portion — only learn from the response
            prompt_len = len(prompt_ids)
            loss_mask  = [0] * min(prompt_len, len(targets)) + \
                         [1] * max(0, len(targets) - prompt_len)
            loss_mask  = loss_mask[:len(targets)]

            # Pad to max_seq_len
            pad_len   = max_seq_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_id] * max(0, pad_len)
            targets   = targets   + [0]               * max(0, pad_len)
            loss_mask = loss_mask + [0]               * max(0, pad_len)

            input_ids = input_ids[:max_seq_len]
            targets   = targets[:max_seq_len]
            loss_mask = loss_mask[:max_seq_len]

            self.examples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(targets,   dtype=torch.long),
                torch.tensor(loss_mask, dtype=torch.float),
            ))

        print(f"   Ready: {len(self.examples):,} examples ({skipped} skipped)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_finetune_data(data_dir: str = "datasets/finetune") -> List[Dict]:
    """
    Load fine-tuning data from JSONL files in data_dir.
    Each file should have one JSON object per line.
    """
    data_path = Path(data_dir)
    all_data  = []

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created {data_dir}/ — add your JSONL data files here")
        return []

    for jsonl_file in sorted(data_path.glob("*.jsonl")):
        file_data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        file_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"   Loaded {len(file_data):,} examples from {jsonl_file.name}")
        all_data.extend(file_data)

    return all_data


def finetune(model, tokenizer, cfg, checkpoint_path: str = "checkpoints/best.pt"):
    """
    Fine-tune Ember on instruction data.

    Args:
        model:           EmberModel instance
        tokenizer:       EmberTokenizer instance
        cfg:             EmberConfig instance
        checkpoint_path: Base model to fine-tune from
    """
    print("\n" + "=" * 52)
    print("  🔥 EMBER — Fine-tuning")
    print("=" * 52)

    # ── Load base model ────────────────────────────────────
    print(f"\n  Loading base model from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model)
    model = model.to(cfg.device)
    model.train()

    # ── Load data ──────────────────────────────────────────
    print("\n  Loading fine-tuning data...")
    raw_data = load_finetune_data("datasets/finetune")

    if not raw_data:
        print("\n  ⚠ No fine-tuning data found!")
        print("  Run: python scripts/download_finetune_data.py")
        print("  Then retry fine-tuning.")
        return

    # Shuffle data
    random.shuffle(raw_data)

    # Split 90/10 train/val
    split     = int(len(raw_data) * 0.9)
    train_data = raw_data[:split]
    val_data   = raw_data[split:]

    print(f"\n  Train examples : {len(train_data):,}")
    print(f"  Val examples   : {len(val_data):,}")

    # ── Build datasets ─────────────────────────────────────
    print("\n  Building training dataset...")
    train_dataset = FinetuneDataset(train_data, tokenizer, cfg.max_seq_len)

    print("\n  Building validation dataset...")
    val_dataset = FinetuneDataset(val_data, tokenizer, cfg.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=0
    )

    # ── Optimizer — lower LR for fine-tuning ──────────────
    # Fine-tuning uses 10x lower LR than pretraining
    # We're nudging, not relearning
    from torch.optim import AdamW
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.learning_rate * 0.1,  # 3e-5
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.95),
    )

    total_steps = len(train_loader) * 3  # 3 epochs of fine-tuning
    scheduler   = build_scheduler(optimizer, cfg, total_steps)

    # ── Fine-tuning loop ───────────────────────────────────
    best_val_loss = float("inf")
    global_step   = 0

    print(f"\n  Total fine-tune steps : {total_steps:,}")
    print(f"  Fine-tune epochs      : 3")
    print(f"  Learning rate         : {cfg.learning_rate * 0.1:.1e}")
    print("\n" + "─" * 52)

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        t_start      = time.time()

        for batch_idx, (input_ids, targets, loss_mask) in enumerate(train_loader):
            input_ids = input_ids.to(cfg.device)
            targets   = targets.to(cfg.device)
            loss_mask = loss_mask.to(cfg.device)

            # Forward pass
            logits, _ = model(input_ids)

            # Masked loss — only on response tokens
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size),
                targets.view(-1),
                reduction="none",
            )
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1)

            # Backward
            (loss / cfg.grad_accumulation).backward()
            running_loss += loss.item()

            if (batch_idx + 1) % cfg.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg_loss    = running_loss / (cfg.log_every * cfg.grad_accumulation)
                    elapsed     = time.time() - t_start
                    print(
                        f"  epoch {epoch+1} | step {global_step:>5,} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0
                    t_start      = time.time()

        # ── Epoch validation ───────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_count = 0

        with torch.no_grad():
            for input_ids, targets, loss_mask in val_loader:
                input_ids = input_ids.to(cfg.device)
                targets   = targets.to(cfg.device)
                loss_mask = loss_mask.to(cfg.device)

                logits, _ = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    targets.view(-1),
                    reduction="none",
                )
                loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1)
                val_loss  += loss.item()
                val_count += 1

        val_loss /= max(1, val_count)
        is_best   = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        print(f"\n  ── Epoch {epoch+1} complete ──")
        print(f"  val_loss   : {val_loss:.4f}")
        print(f"  perplexity : {math.exp(min(val_loss, 20)):.1f}")

        # Save fine-tuned checkpoint
        save_checkpoint(
            model, optimizer, scheduler,
            step      = global_step,
            epoch     = epoch,
            loss      = val_loss,
            val_loss  = val_loss,
            cfg       = cfg,
            is_best   = is_best,
        )

        # Also save dedicated finetuned checkpoint
        ft_path = Path("checkpoints") / f"finetuned_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "val_loss": val_loss}, ft_path)
        print(f"  💾 Saved → {ft_path.name}\n")

    print("=" * 52)
    print("  Fine-tuning complete!")
    print(f"  Best val_loss : {best_val_loss:.4f}")
    print("=" * 52)
