from __future__ import annotations

import time

import torch
import torch.nn as nn


class TrainingBenchmarker:
    def __init__(self, model: nn.Module, tokens_per_iter: int, device: torch.device):
        self.model = model
        self.tokens_per_iter = tokens_per_iter
        self.device = device
        self.iter_times: list[float] = []
        self._iter_start: float | None = None
        self.train_start: float | None = None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def iter_start(self) -> None:
        self._iter_start = time.perf_counter()

    def iter_end(self) -> None:
        if self._iter_start is not None:
            self.iter_times.append(time.perf_counter() - self._iter_start)

    def peak_memory_mb(self) -> float | None:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1e6
        if self.device.type == "mps":
            try:
                return torch.mps.current_allocated_memory() / 1e6
            except AttributeError:
                return None
        return None

    def summary(self) -> dict[str, float]:
        total_time = time.perf_counter() - (self.train_start or time.perf_counter())
        n_iters = len(self.iter_times)
        avg_iter_s = sum(self.iter_times) / n_iters if n_iters else 0.0
        throughput = self.tokens_per_iter / avg_iter_s if avg_iter_s else 0.0
        return {
            "params": float(self.count_params()),
            "total_time": total_time,
            "iterations": float(n_iters),
            "avg_iter_ms": avg_iter_s * 1000.0,
            "throughput_toks": throughput,
        }

    def print_summary(self) -> None:
        s = self.summary()
        width = 60
        sep = "-" * width
        print(f"\n{'=' * width}")
        print(" TRAINING BENCHMARK SUMMARY")
        print(f"{'=' * width}")
        print(f" {'Parameters':<26} {int(s['params']):>18,}")
        print(sep)
        print(f" {'Total training time (s)':<26} {s['total_time']:>18.2f}")
        print(f" {'Iterations':<26} {int(s['iterations']):>18,}")
        print(f" {'Avg iter (ms)':<26} {s['avg_iter_ms']:>18.2f}")
        print(f" {'Throughput tok/s':<26} {s['throughput_toks']:>18,.0f}")
        mem = self.peak_memory_mb()
        if mem is not None:
            print(f" {'Peak memory MB':<26} {mem:>18.1f}")
        print(f"{'=' * width}\n")
