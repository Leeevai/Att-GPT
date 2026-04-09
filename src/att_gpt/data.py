from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import CharTokenizer


@dataclass(slots=True)
class Corpus:
    text: str
    tokenizer: CharTokenizer
    train_data: torch.Tensor
    val_data: torch.Tensor


def build_corpus(path: Path, train_split: float = 0.9) -> Corpus:
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(encoded))
    return Corpus(text=text, tokenizer=tokenizer, train_data=encoded[:n], val_data=encoded[n:])


def get_batch(
    source: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
