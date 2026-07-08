# ============================================================
#  EMBER - Tokenizer
#  Wraps BPE with clean encode() / decode() interface
# ============================================================

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .bpe import BPETrainer, get_pre_tokens


class EmberTokenizer:
    """
    Converts text <-> token IDs using trained BPE merges.

    Usage:
        tok  = EmberTokenizer.load("tokenizer/ember.tokenizer.json")
        ids  = tok.encode("Hello world")   # → [234, 891, 12]
        text = tok.decode([234, 891, 12])  # → "Hello world"
    """

    def __init__(self, trainer: BPETrainer):
        self.trainer       = trainer
        self.vocab         = trainer.vocab
        self.inverse_vocab = trainer.inverse_vocab
        self.merges        = trainer.merges

        self.merge_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        self.pad_id = self.vocab.get("<pad>", 0)
        self.unk_id = self.vocab.get("<unk>", 1)
        self.bos_id = self.vocab.get("<bos>", 2)
        self.eos_id = self.vocab.get("<eos>", 3)

    def _encode_word(self, word: str) -> List[int]:
        tokens = list(word) + ["</w>"]

        while len(tokens) > 1:
            best_rank = float("inf")
            best_idx  = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx  = i

            if best_idx == -1 or best_rank == float("inf"):
                break

            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        return [self.vocab.get(tok, self.unk_id) for tok in tokens]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids: List[int] = []

        if add_bos:
            ids.append(self.bos_id)

        for pre_token in get_pre_tokens(text):
            ids.extend(self._encode_word(pre_token))

        if add_eos:
            ids.append(self.eos_id)

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        pad: bool = True,
    ) -> List[List[int]]:
        encoded = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            for t in texts
        ]
        if pad and encoded:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        return encoded

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        if not skip_special_tokens:
            special_ids = set()

        tokens = []
        for id_ in ids:
            if id_ in special_ids:
                continue
            tokens.append(self.inverse_vocab.get(id_, "<unk>"))

        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = re.sub(r"(?<!\n) +", " ", text).strip()  # preserve newlines
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, id_: int) -> str:
        return self.inverse_vocab.get(id_, "<unk>")

    def save(self, path: str) -> None:
        self.trainer.save(path)

    @classmethod
    def load(cls, path: str) -> "EmberTokenizer":
        trainer = BPETrainer.load(path)
        return cls(trainer)

    def __repr__(self) -> str:
        return f"EmberTokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
