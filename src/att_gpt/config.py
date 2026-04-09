from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    @property
    def head_size(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        return self.n_embd // self.n_head


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 1337
    output_dir: Path = Path("checkpoints")
