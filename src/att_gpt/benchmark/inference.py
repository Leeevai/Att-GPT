from __future__ import annotations

import time

import torch
import torch.nn as nn


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    max_new_tokens: int = 200,
    warmup: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.eval()
    start_token = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 20)

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(start_token, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - t0

    stats = {
        "tokens": float(max_new_tokens),
        "elapsed_s": elapsed,
        "tokens_per_s": max_new_tokens / elapsed if elapsed else 0.0,
        "ms_per_token": (elapsed * 1000.0) / max_new_tokens if max_new_tokens else 0.0,
    }
    model.train()
    return out, stats
