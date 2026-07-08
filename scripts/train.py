# ============================================================
#  EMBER - Launch Training
#  Run from ember/ root:
#    python scripts/train.py           # fresh start
#    python scripts/train.py --resume  # resume from checkpoint
# ============================================================

import sys
import argparse
sys.path.insert(0, ".")

from config import cfg
from model.ember_model import build_ember
from tokenizer.tokenizer import EmberTokenizer
from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train Ember")
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from latest checkpoint (default: True)"
    )
    parser.add_argument(
        "--fresh", action="store_true", default=False,
        help="Start training from scratch, ignore checkpoints"
    )
    args = parser.parse_args()

    resume = args.resume and not args.fresh

    # Print config
    cfg.summary()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = EmberTokenizer.load(cfg.tokenizer_file)
    print(f"Tokenizer: {tokenizer}\n")

    # Build model
    print("Building model...")
    model = build_ember(cfg)
    model.param_summary()

    # Train
    train(model, tokenizer, cfg, resume=resume)


if __name__ == "__main__":
    main()
