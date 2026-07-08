# 🔥 Ember — Built From Scratch

> *"Small now. The start of a fire."*

Ember is a small language model built entirely from scratch — no pretrained weights, no shortcuts.  
Every component: tokenizer, architecture, training loop, and inference engine — written by hand.

---

## Roadmap

| Version | Hardware | Parameters | Context | Status |
|---------|----------|------------|---------|--------|
| **Ember v1** | 8GB RAM, CPU | 50–125M | 512 tokens | 🔨 In Progress |
| **Flame v2** | 16GB RAM, CPU | 125–350M | 1024–2048 tokens | 📋 Planned |

---

## Project Structure

```
ember/
├── config.py           ← Single source of truth for all hyperparameters
├── requirements.txt    ← Dependencies
│
├── tokenizer/          ← BPE tokenizer (built from scratch)
│   ├── bpe.py          ← Byte-pair encoding algorithm
│   ├── train_tokenizer.py
│   └── tokenizer.py    ← Tokenizer class (encode/decode)
│
├── model/              ← Transformer architecture
│   ├── attention.py    ← Multi-head self-attention
│   ├── feedforward.py  ← Feed-forward blocks
│   ├── transformer.py  ← Full transformer stack
│   └── ember_model.py  ← Top-level model class
│
├── data/               ← Dataset pipeline
│   ├── stream.py       ← Streaming dataset loader (no OOM)
│   └── preprocess.py   ← Tokenization + batching
│
├── training/           ← Training engine
│   ├── trainer.py      ← Main training loop
│   ├── optimizer.py    ← AdamW + LR scheduler
│   └── checkpoint.py   ← Save / resume training
│
├── inference/          ← Text generation
│   └── generate.py     ← Sampling, top-k, top-p
│
├── scripts/            ← Runnable scripts
│   ├── train.py        ← Start / resume training
│   ├── chat.py         ← Talk to Ember
│   └── eval.py         ← Evaluate perplexity
│
├── checkpoints/        ← Saved model weights (auto-created)
├── logs/               ← Training logs (auto-created)
├── datasets/           ← Cached dataset files (auto-created)
└── tests/              ← Unit tests
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the tokenizer on your dataset
python scripts/train_tokenizer.py

# 3. Start training Ember
python scripts/train.py

# 4. Chat with Ember
python scripts/chat.py --checkpoint checkpoints/latest.pt
```

---

## Design Principles

- **Memory-safe**: Streaming datasets, gradient accumulation, checkpointing — never crashes
- **Pausable**: Training can stop and resume at any checkpoint
- **Transparent**: Every file is readable, commented, no magic
- **Upgradeable**: Config-driven — scaling to Flame v2 is just changing numbers in `config.py`

---

## Built With

- PyTorch (CPU)
- HuggingFace `datasets` (streaming)
- HuggingFace `tokenizers` (BPE)
- Pure math and patience
