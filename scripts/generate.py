from __future__ import annotations

import argparse
from pathlib import Path

import torch

from att_gpt.config import ModelConfig
from att_gpt.device import get_device
from att_gpt.model import GPTLanguageModel
from att_gpt.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained att-gpt checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    p.add_argument("--tokens", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = payload["model_config"]
    model_cfg = ModelConfig(**cfg_dict)
    model = GPTLanguageModel(model_cfg)
    model.load_state_dict(payload["state_dict"])

    device = get_device()
    model = model.to(device)
    model.eval()

    tokenizer = CharTokenizer.load(args.tokenizer)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.tokens)

    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
