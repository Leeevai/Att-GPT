from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .benchmark import TrainingBenchmarker, benchmark_inference
from .config import ModelConfig, TrainConfig
from .data import build_corpus, get_batch
from .device import get_device
from .model import GPTLanguageModel


@dataclass(slots=True)
class TrainArtifacts:
    model: GPTLanguageModel
    model_config: ModelConfig
    train_config: TrainConfig
    tokenizer_path: Path
    checkpoint_path: Path
    device: torch.device


def estimate_loss(
    model: GPTLanguageModel,
    corpus_train: torch.Tensor,
    corpus_val: torch.Tensor,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    device: torch.device,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for split_name, source in (("train", corpus_train), ("val", corpus_val)):
            losses = []
            for _ in range(train_cfg.eval_iters):
                xb, yb = get_batch(source, train_cfg.batch_size, model_cfg.block_size, device)
                _, loss = model(xb, yb)
                losses.append(float(loss.item()))
            out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


def train(data_path: Path, model_cfg: ModelConfig, train_cfg: TrainConfig) -> TrainArtifacts:
    device = get_device()
    torch.manual_seed(train_cfg.seed)

    corpus = build_corpus(data_path)
    model_cfg = ModelConfig(
        vocab_size=corpus.tokenizer.vocab_size,
        block_size=model_cfg.block_size,
        n_embd=model_cfg.n_embd,
        n_head=model_cfg.n_head,
        n_layer=model_cfg.n_layer,
        dropout=model_cfg.dropout,
    )

    model = GPTLanguageModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    tokens_per_iter = train_cfg.batch_size * model_cfg.block_size
    bm = TrainingBenchmarker(model=model, tokens_per_iter=tokens_per_iter, device=device)

    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = train_cfg.output_dir / "tokenizer.json"
    checkpoint_path = train_cfg.output_dir / "model.pt"
    corpus.tokenizer.save(tokenizer_path)

    print(f"Using device: {device}")
    print(f"Model parameters: {bm.count_params():,}")

    bm.train_start = __import__("time").perf_counter()
    for it in range(train_cfg.max_iters):
        if it % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, corpus.train_data, corpus.val_data, train_cfg, model_cfg, device)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        bm.iter_start()
        xb, yb = get_batch(corpus.train_data, train_cfg.batch_size, model_cfg.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bm.iter_end()

    bm.print_summary()
    out, stats = benchmark_inference(model=model, device=device, max_new_tokens=200)
    print(f"Inference: {stats['tokens_per_s']:.1f} tok/s | {stats['ms_per_token']:.2f} ms/token")
    sample = corpus.tokenizer.decode(out[0].tolist())
    print("--- Sample text ---")
    print(sample)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": {
                **asdict(train_cfg),
                "output_dir": str(train_cfg.output_dir),
            },
        },
        checkpoint_path,
    )

    return TrainArtifacts(
        model=model,
        model_config=model_cfg,
        train_config=train_cfg,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
