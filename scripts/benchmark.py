from __future__ import annotations

import argparse
from pathlib import Path

import torch

from att_gpt.benchmark import benchmark_inference
from att_gpt.config import ModelConfig
from att_gpt.device import get_device
from att_gpt.model import GPTLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference benchmark for a saved checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--tokens", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**payload["model_config"])
    model = GPTLanguageModel(model_cfg)
    model.load_state_dict(payload["state_dict"])

    device = get_device()
    model = model.to(device)
    _, stats = benchmark_inference(model=model, device=device, max_new_tokens=args.tokens)
    print(f"Tokens: {int(stats['tokens'])}")
    print(f"Elapsed: {stats['elapsed_s']:.3f}s")
    print(f"Throughput: {stats['tokens_per_s']:.1f} tok/s")
    print(f"Latency: {stats['ms_per_token']:.2f} ms/token")


if __name__ == "__main__":
    main()
