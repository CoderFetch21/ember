# ============================================================
#  EMBER - Byte-Pair Encoding (BPE) Algorithm
#  Built from scratch — no black boxes
# ============================================================
#
#  HOW BPE WORKS (plain English):
#
#  1. Start with every character as its own token
#     "hello" → ["h", "e", "l", "l", "o"]
#
#  2. Count every pair of adjacent tokens across all training text
#     ("h","e") appears 500x, ("e","l") appears 800x, etc.
#
#  3. Merge the most frequent pair into a new token
#     ("e","l") → "el"   ...now vocab has "el" as a token
#
#  4. Repeat steps 2-3 until vocab reaches target size (16,000)
#
# ============================================================

import re
import json
import collections
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def get_pre_tokens(text: str) -> List[str]:
    return re.findall(GPT2_PATTERN, text)


def text_to_char_sequences(text: str) -> Dict[Tuple[str, ...], int]:
    word_freqs: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
    for pre_token in get_pre_tokens(text):
        chars = tuple(list(pre_token) + ["</w>"])
        word_freqs[chars] += 1
    return dict(word_freqs)


def get_pair_frequencies(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    pairs: Dict[Tuple[str, str], int] = collections.defaultdict(int)
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pairs[(word_tokens[i], word_tokens[i + 1])] += freq
    return dict(pairs)


def merge_pair(vocab: Dict[Tuple[str, ...], int], pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
    new_vocab: Dict[Tuple[str, ...], int] = {}
    merged = pair[0] + pair[1]
    for word_tokens, freq in vocab.items():
        new_tokens: List[str] = []
        i = 0
        while i < len(word_tokens):
            if (i < len(word_tokens) - 1
                    and word_tokens[i] == pair[0]
                    and word_tokens[i + 1] == pair[1]):
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab


class BPETrainer:
    """
    Trains a BPE tokenizer on raw text.

    Usage:
        trainer = BPETrainer(vocab_size=16_000)
        trainer.train(texts)
        trainer.save("ember.tokenizer.json")
    """

    def __init__(self, vocab_size: int = 16_000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

    def _build_base_vocab(self, word_freqs: Dict[Tuple[str, ...], int]) -> Dict[str, int]:
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        vocab = {tok: i for i, tok in enumerate(special_tokens)}
        chars = set()
        for word_tokens in word_freqs.keys():
            for char in word_tokens:
                chars.add(char)
        for char in sorted(chars):
            if char not in vocab:
                vocab[char] = len(vocab)
        return vocab

    def train(self, texts: List[str], verbose: bool = True) -> None:
        if verbose:
            print("Ember BPE Trainer starting...")
            print(f"   Target vocab size : {self.vocab_size}")
            print(f"   Input texts       : {len(texts)}")

        word_freqs: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
        for text in tqdm(texts, desc="   Scanning text", disable=not verbose):
            for token_seq, freq in text_to_char_sequences(text).items():
                word_freqs[token_seq] += freq

        if verbose:
            print(f"   Unique word types : {len(word_freqs)}")

        self.vocab = self._build_base_vocab(word_freqs)
        base_size = len(self.vocab)

        if verbose:
            print(f"   Base vocab size   : {base_size} characters")
            print(f"   Merges to learn   : {self.vocab_size - base_size}")

        num_merges = self.vocab_size - base_size
        current_vocab = dict(word_freqs)

        pbar = tqdm(range(num_merges), desc="   Learning merges", disable=not verbose)
        for i in pbar:
            pairs = get_pair_frequencies(current_vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            if best_freq < 2:
                break

            current_vocab = merge_pair(current_vocab, best_pair)
            self.merges.append(best_pair)

            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if verbose and i % 500 == 0:
                pbar.set_postfix({
                    "vocab": len(self.vocab),
                    "best": f"{best_pair[0]}+{best_pair[1]}",
                    "freq": best_freq
                })

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"\nTraining complete!")
            print(f"   Final vocab size  : {len(self.vocab)}")
            print(f"   Total merges      : {len(self.merges)}")

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "special_tokens": {
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<bos>",
                "eos": "<eos>",
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETrainer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trainer = cls(vocab_size=data["vocab_size"])
        trainer.vocab = data["vocab"]
        trainer.merges = [tuple(m) for m in data["merges"]]
        trainer.inverse_vocab = {v: k for k, v in trainer.vocab.items()}
        return trainer
