# ============================================================
#  EMBER - DataLoader Factory
# ============================================================

from torch.utils.data import DataLoader
from .stream import StreamingTextDataset, ValidationDataset


def get_train_loader(tokenizer, cfg) -> DataLoader:
    """
    Returns the training DataLoader.
    Streams data — safe for 8GB RAM.
    """
    dataset = StreamingTextDataset(tokenizer, cfg, split="train")

    return DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        num_workers = 0,     # 0 required for streaming datasets
        pin_memory  = False, # CPU only
    )


def get_val_loader(tokenizer, cfg, max_samples: int = 500) -> DataLoader:
    """
    Returns the validation DataLoader.
    Capped at max_samples to keep eval fast during training.
    """
    dataset = ValidationDataset(tokenizer, cfg, max_samples=max_samples)

    return DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        num_workers = 0,
        pin_memory  = False,
    )
