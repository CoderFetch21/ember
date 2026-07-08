# ============================================================
#  EMBER - Launch Fine-tuning
#  Run from ember/ root:
#    python scripts/finetune.py
# ============================================================

import sys
sys.path.insert(0, ".")

from config import cfg
from model.ember_model import build_ember
from tokenizer.tokenizer import EmberTokenizer
from training.finetune import finetune


def main():
    cfg.summary()

    print("Loading tokenizer...")
    tokenizer = EmberTokenizer.load(cfg.tokenizer_file)
    print(f"Tokenizer: {tokenizer}\n")

    print("Building model...")
    model = build_ember(cfg)
    model.param_summary()

    # Fine-tune from best pretrained checkpoint
    finetune(model, tokenizer, cfg, checkpoint_path="checkpoints/best.pt")


if __name__ == "__main__":
    main()
