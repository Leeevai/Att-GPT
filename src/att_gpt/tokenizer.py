from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        payload = {"stoi": self.stoi}
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = payload["stoi"]
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)
