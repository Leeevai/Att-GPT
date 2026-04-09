from __future__ import annotations

import argparse
from pathlib import Path

from att_gpt.config import ModelConfig, TrainConfig
from att_gpt.trainer import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train att-gpt on Shakespeare text.")
    p.add_argument("--data-path", type=Path, default=Path("data/raw/shakespeare.txt"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig(
        vocab_size=1,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    train(args.data_path, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
