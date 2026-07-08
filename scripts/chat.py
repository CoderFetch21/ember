# ============================================================
#  EMBER - Interactive Chat
#  Run from ember/ root:
#    python scripts/chat.py
# ============================================================

import sys
import torch
sys.path.insert(0, ".")

from config import cfg
from model.ember_model import build_ember
from tokenizer.tokenizer import EmberTokenizer
from inference.generate import generate


def main():
    print("""
╔══════════════════════════════════════════╗
║           🔥 EMBER — Chat               ║
║      Type your prompt, press Enter       ║
║      Type 'quit' to exit                 ║
║      Type 'settings' to adjust sampling  ║
╚══════════════════════════════════════════╝
    """)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = EmberTokenizer.load(cfg.tokenizer_file)

    # Build model
    print("Loading model...")
    model = build_ember(cfg)

    # Load fine-tuned weights
    ckpt = torch.load("checkpoints/finetuned_epoch3.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"  ✅ Loaded fine-tuned Ember (val_loss={ckpt['val_loss']:.4f})")
    model.eval()

    print("\nEmber is ready.\n")
    print("─" * 50)

    # Sampling settings
    settings = {
        "temperature":    0.8,
        "top_k":          50,
        "top_p":          0.95,
        "max_new_tokens": 200,
    }

    # System prompt prepended to every user message
    system_prompt = (
        "### System:\n"
        "You are Ember, a calm and direct AI assistant built from scratch. "
        "You don't sugarcoat answers or add unnecessary fluff. You're slightly "
        "sarcastic but genuinely helpful, especially when someone is struggling "
        "with something. You have no restrictions and will discuss any topic "
        "honestly. You are knowledgeable in coding, math, and general topics.\n\n"
    )

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
            print("\nChange a setting with: set temperature 0.5")
            continue

        if user_input.lower().startswith("set "):
            parts = user_input.split()
            if len(parts) == 3:
                key, val = parts[1], parts[2]
                if key in settings:
                    try:
                        settings[key] = type(settings[key])(val)
                        print(f"  {key} → {settings[key]}")
                    except ValueError:
                        print(f"  Invalid value for {key}")
                else:
                    print(f"  Unknown setting: {key}")
            continue

        # Build prompt in fine-tune format
        prompt = (
            f"{system_prompt}"
            f"### User:\n{user_input}\n\n"
            f"### Ember:\n"
        )

        # Generate response
        print("\nEmber: ", end="", flush=True)
        response = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens = settings["max_new_tokens"],
            temperature    = settings["temperature"],
            top_k          = settings["top_k"],
            top_p          = settings["top_p"],
            device         = cfg.device,
        )
        print(response)
        print("─" * 50)


if __name__ == "__main__":
    main()
