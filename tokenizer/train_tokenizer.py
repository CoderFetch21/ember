# ============================================================
#  EMBER - Train the Tokenizer
#  Run this ONCE before training the model
#
#  Usage:
#    python -m tokenizer.train_tokenizer
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import cfg
from tokenizer.bpe import BPETrainer
from tokenizer.tokenizer import EmberTokenizer


def stream_training_texts(num_samples: int = 50_000) -> list:
    print(f"Streaming {num_samples:,} samples from Wikipedia...")
    print("(This streams — nothing large is downloaded at once)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    texts = []
    for i, sample in enumerate(dataset):
        text = sample.get("text", "").strip()
        if text and len(text) > 50:
            texts.append(text)
        if len(texts) >= num_samples:
            break
        if i % 10_000 == 0 and i > 0:
            print(f"   Collected {len(texts):,} samples so far...")

    print(f"   Done. Collected {len(texts):,} text samples.\n")
    return texts


def main():
    print("=" * 50)
    print("  EMBER — Tokenizer Training")
    print("=" * 50)
    print()

    tokenizer_path = cfg.tokenizer_file
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer already exists at: {tokenizer_path}")
        resp = input("Retrain from scratch? [y/N]: ").strip().lower()
        if resp != "y":
            print("Keeping existing tokenizer.")
            tok = EmberTokenizer.load(tokenizer_path)
            print(f"\nLoaded: {tok}")
            test_encode_decode(tok)
            return

    texts = stream_training_texts(num_samples=50_000)

    trainer = BPETrainer(vocab_size=cfg.vocab_size)
    trainer.train(texts, verbose=True)

    tokenizer = EmberTokenizer(trainer)
    tokenizer.save(tokenizer_path)

    print(f"\nFinal tokenizer: {tokenizer}")
    test_encode_decode(tokenizer)


def test_encode_decode(tok: EmberTokenizer):
    print("\n--- Sanity Check ---")
    test_cases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "What is 2 + 2?",
    ]
    for text in test_cases:
        ids     = tok.encode(text)
        decoded = tok.decode(ids)
        match   = decoded.strip() == text.strip()
        status  = "PASS" if match else "DIFF"
        print(f"  [{status}] '{text[:40]}' → {len(ids)} tokens")


if __name__ == "__main__":
    main()
