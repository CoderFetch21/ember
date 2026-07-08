# ============================================================
#  EMBER - Streaming Dataset Pipeline
# ============================================================
#
#  WHAT THIS DOES (plain English):
#
#  We can't load the entire Wikipedia + books dataset into RAM —
#  it would be hundreds of gigabytes. Instead we STREAM it:
#  fetch one chunk at a time, tokenize it, yield batches, repeat.
#
#  The pipeline:
#    HuggingFace dataset (streaming)
#        ↓
#    Raw text chunks
#        ↓
#    Tokenize → list of token IDs
#        ↓
#    Pack into fixed-length sequences (512 tokens each)
#        ↓
#    Yield (input_ids, targets) tensor pairs
#
#  "Packing" means we concatenate all tokens into one long stream
#  then cut into 512-token chunks. No wasted padding, maximum
#  efficiency. This is how GPT models are trained.
#
# ============================================================

import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Tuple, List
from datasets import load_dataset


class StreamingTextDataset(IterableDataset):
    """
    Streams text from HuggingFace datasets, tokenizes on the fly,
    and yields fixed-length (input, target) tensor pairs.

    Never loads the full dataset into memory.

    Usage:
        dataset = StreamingTextDataset(tokenizer, cfg, split="train")
        for input_ids, targets in DataLoader(dataset, batch_size=4):
            loss = model(input_ids, targets)
    """

    def __init__(self, tokenizer, cfg, split: str = "train"):
        super().__init__()
        self.tokenizer      = tokenizer
        self.seq_len        = cfg.max_seq_len
        self.dataset_name   = cfg.dataset_name
        self.dataset_config = cfg.dataset_config
        self.split          = split
        self.bos_id         = tokenizer.bos_id
        self.eos_id         = tokenizer.eos_id

    def _token_stream(self) -> Iterator[int]:
        """
        Yields individual token IDs from the dataset, one at a time.
        This is the raw stream before packing into sequences.
        """
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
            trust_remote_code=False,
        )

        for sample in dataset:
            text = sample.get("text", "").strip()
            if not text or len(text) < 20:
                continue

            # Encode with BOS/EOS markers around each document
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)

            for id_ in ids:
                yield id_

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Yields (input_ids, targets) pairs of shape (seq_len,).

        input_ids: tokens  0 .. seq_len-1
        targets:   tokens  1 .. seq_len     (shifted by 1 — next token prediction)

        Example with seq_len=5:
            stream:     [12, 34, 56, 78, 90, 11, ...]
            input_ids:  [12, 34, 56, 78, 90]
            targets:    [34, 56, 78, 90, 11]
        """
        buffer: List[int] = []

        for token_id in self._token_stream():
            buffer.append(token_id)

            # Once we have seq_len + 1 tokens, yield a training pair
            if len(buffer) == self.seq_len + 1:
                input_ids = torch.tensor(buffer[:-1], dtype=torch.long)
                targets   = torch.tensor(buffer[1:],  dtype=torch.long)
                yield input_ids, targets

                # Slide forward — keep last token as start of next sequence
                buffer = buffer[self.seq_len:]


class ValidationDataset(IterableDataset):
    """
    Same as StreamingTextDataset but for validation.
    Caps at max_samples to keep eval fast.
    """

    def __init__(self, tokenizer, cfg, max_samples: int = 500):
        super().__init__()
        self.tokenizer      = tokenizer
        self.seq_len        = cfg.max_seq_len
        self.dataset_name   = cfg.dataset_name
        self.dataset_config = cfg.dataset_config
        self.max_samples    = max_samples

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="validation",
            streaming=True,
            trust_remote_code=False,
        )

        buffer: List[int] = []
        yielded = 0

        for sample in dataset:
            if yielded >= self.max_samples:
                break

            text = sample.get("text", "").strip()
            if not text or len(text) < 20:
                continue

            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len + 1 and yielded < self.max_samples:
                input_ids = torch.tensor(buffer[:self.seq_len],      dtype=torch.long)
                targets   = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                yield input_ids, targets
                buffer = buffer[self.seq_len:]
                yielded += 1
