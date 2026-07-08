# ============================================================
#  EMBER - Text Generation Engine
# ============================================================
#
#  WHAT THIS DOES (plain English):
#
#  Takes a prompt, feeds it to Ember, and repeatedly samples
#  the next token until we hit max_new_tokens or <eos>.
#
#  SAMPLING STRATEGIES:
#
#  Temperature:
#    Controls randomness. Low (0.5) = focused/repetitive.
#    High (1.2) = creative/chaotic. 0.8 is the sweet spot.
#
#  Top-K:
#    Only consider the K most likely next tokens.
#    Prevents Ember from picking totally nonsense tokens.
#
#  Top-P (nucleus sampling):
#    Only consider tokens whose cumulative probability
#    reaches P. More dynamic than top-k — automatically
#    narrows when the model is confident, widens when unsure.
#
#  We apply all three in order: temperature → top-k → top-p
#
# ============================================================

import torch
import torch.nn.functional as F
from typing import Optional, List


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out logits outside the nucleus (top-p) set."""
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once cumulative prob exceeds p
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter back to original order
    logits = torch.scatter(logits, -1, sorted_indices, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt.

    Args:
        model:          Trained EmberModel
        tokenizer:      EmberTokenizer
        prompt:         Input text to continue from
        max_new_tokens: Maximum tokens to generate
        temperature:    Sampling temperature (0.5=focused, 1.2=creative)
        top_k:          Top-k filtering (0 to disable)
        top_p:          Nucleus sampling threshold (1.0 to disable)
        device:         "cpu" or "cuda"

    Returns:
        Generated text (not including the prompt)
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True)
    if not input_ids:
        input_ids = [tokenizer.bos_id]

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated: List[int] = []

    for _ in range(max_new_tokens):
        # Truncate context if it exceeds max_seq_len
        context = ids[:, -model.max_seq_len:]

        # Forward pass — only need logits for the last position
        logits, _ = model(context)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Apply top-k
        next_logits = top_k_filter(next_logits, top_k)

        # Apply top-p
        next_logits = top_p_filter(next_logits, top_p)

        # Sample from distribution
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        # Stop if we hit end-of-sequence
        if next_id == tokenizer.eos_id:
            break

        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated)
