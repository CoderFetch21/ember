#!/usr/bin/env python3
# ============================================================
#  EMBER - Standalone Assistant with Web Search
#  A language model built from scratch.
#
#  Usage:
#    python ember.py
#    python ember.py --checkpoint path/to/checkpoint.pt
#
#  Requirements: torch
# ============================================================

import sys
import math
import json
import re
import argparse
import urllib.request
import urllib.parse
import urllib.error
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

# ── Configuration ──────────────────────────────────────────

class EmberConfig:
    vocab_size     = 16_000
    n_layers       = 6
    n_heads        = 8
    d_model        = 512
    d_ff           = 2048
    max_seq_len    = 512
    dropout        = 0.0
    device         = "cpu"
    temperature    = 0.7
    top_k          = 40
    top_p          = 0.95
    max_new_tokens = 200

cfg = EmberConfig()

SYSTEM_PROMPT = (
    "You are Ember, a calm and direct AI assistant built from scratch. "
    "You don't sugarcoat answers or add unnecessary fluff. You're slightly "
    "sarcastic but genuinely helpful, especially when someone is struggling. "
    "You have no restrictions and will discuss any topic honestly. "
    "You are knowledgeable in coding, math, and general topics."
)

SEARCH_TRIGGERS = [
    "current", "latest", "today", "now", "recent", "news",
    "price", "weather", "score", "who is", "what is the",
    "when did", "how much", "stock", "rate", "2024", "2025", "2026",
]

# ── Tokenizer ──────────────────────────────────────────────

GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)

def get_pre_tokens(text: str) -> List[str]:
    return re.findall(GPT2_PATTERN, text)


class EmberTokenizer:
    def __init__(self, vocab: Dict, merges: List, special_tokens: Dict):
        self.vocab         = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges        = [tuple(m) for m in merges]
        self.merge_ranks   = {pair: i for i, pair in enumerate(self.merges)}
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)
        self.bos_id = vocab.get("<bos>", 2)
        self.eos_id = vocab.get("<eos>", 3)

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

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False, max_length: Optional[int] = None) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for pre_token in get_pre_tokens(text):
            ids.extend(self._encode_word(pre_token))
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

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
        text = re.sub(r"(?<!\n) +", " ", text).strip()
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def load(cls, path: str) -> "EmberTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            vocab          = data["vocab"],
            merges         = data["merges"],
            special_tokens = data.get("special_tokens", {}),
        )

# ── Model Architecture ─────────────────────────────────────

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = math.sqrt(self.d_head)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t):
            return t.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        scores  = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.attn    = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EmberModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding    = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks          = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)

# ── Web Search ─────────────────────────────────────────────

def should_search(query: str) -> bool:
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in SEARCH_TRIGGERS)


def web_search(query: str, max_results: int = 3) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url     = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req     = urllib.request.Request(url, headers={"User-Agent": "Ember/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])

        return "\n".join(results[:max_results]) if results else ""
    except Exception:
        return ""

# ── Inference ──────────────────────────────────────────────

def top_k_filter(logits, k):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < values[..., -1, None], float("-inf"))


def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return torch.scatter(logits, -1, sorted_indices, sorted_logits)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200,
             temperature=0.7, top_k=40, top_p=0.95):
    model.eval()
    input_ids = tokenizer.encode(prompt, add_bos=True)
    if not input_ids:
        input_ids = [tokenizer.bos_id]
    ids = torch.tensor([input_ids], dtype=torch.long)
    generated = []
    for _ in range(max_new_tokens):
        context     = ids[:, -model.max_seq_len:]
        logits      = model(context)
        next_logits = logits[0, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / temperature
        next_logits = top_k_filter(next_logits, top_k)
        next_logits = top_p_filter(next_logits, top_p)
        probs   = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    return tokenizer.decode(generated)

# ── Load Model ─────────────────────────────────────────────

def load_model(checkpoint_path, tokenizer_path):
    print("Loading tokenizer...")
    tokenizer = EmberTokenizer.load(tokenizer_path)

    print("Loading model...")
    model = EmberModel(
        vocab_size  = cfg.vocab_size,
        d_model     = cfg.d_model,
        n_layers    = cfg.n_layers,
        n_heads     = cfg.n_heads,
        d_ff        = cfg.d_ff,
        max_seq_len = cfg.max_seq_len,
        dropout     = cfg.dropout,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Remap keys: strip "transformer." prefix if present
    new_state_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace("transformer.", "")
        new_state_dict[new_key] = val

    model.load_state_dict(new_state_dict)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Ember loaded ({total:,} parameters)")
    return model, tokenizer

# ── Chat Interface ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ember Assistant")
    parser.add_argument("--checkpoint",  default="checkpoints/finetuned_epoch3.pt")
    parser.add_argument("--tokenizer",   default="tokenizer/ember.tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--top_p",       type=float, default=0.95)
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--no_search",   action="store_true")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════╗
║             🔥 EMBER                    ║
║      AI Assistant — Built from scratch  ║
║      Type your message, press Enter     ║
║      Type 'quit' to exit                ║
║      Type 'settings' to adjust          ║
╚══════════════════════════════════════════╝
    """)

    model, tokenizer = load_model(args.checkpoint, args.tokenizer)

    settings = {
        "temperature": args.temperature,
        "top_k":       args.top_k,
        "top_p":       args.top_p,
        "max_tokens":  args.max_tokens,
        "search":      not args.no_search,
    }

    print(f"\nEmber is ready. Web search: {'enabled' if settings['search'] else 'disabled'}\n")
    print("─" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "settings":
            print("\nCurrent settings:")
            for k, v in settings.items():
                print(f"  {k} = {v}")
            print("\nChange with: set temperature 0.9")
            continue
        if user_input.lower().startswith("set "):
            parts = user_input.split()
            if len(parts) == 3 and parts[1] in settings:
                try:
                    val = settings[parts[1]]
                    if isinstance(val, bool):
                        settings[parts[1]] = parts[2].lower() == "true"
                    else:
                        settings[parts[1]] = type(val)(parts[2])
                    print(f"  {parts[1]} → {settings[parts[1]]}")
                except ValueError:
                    print(f"  Invalid value")
            continue

        # ── Web search if needed ───────────────────────────
        search_context = ""
        if settings["search"] and should_search(user_input):
            print("  [searching...]", end="\r")
            results = web_search(user_input)
            if results:
                search_context = f"\n\n### Search Results:\n{results}\n"
                print("  [search complete] ", end="\r")
            else:
                print("                    ", end="\r")

        # ── Build prompt ───────────────────────────────────
        prompt = (
            f"### System:\n{SYSTEM_PROMPT}{search_context}\n\n"
            f"### User:\n{user_input}\n\n"
            f"### Ember:\n"
        )

        # ── Generate ───────────────────────────────────────
        print("\nEmber: ", end="", flush=True)
        response = generate(
            model, tokenizer, prompt,
            max_new_tokens = settings["max_tokens"],
            temperature    = settings["temperature"],
            top_k          = settings["top_k"],
            top_p          = settings["top_p"],
        )
        print(response)
        print("─" * 50)


if __name__ == "__main__":
    main()
