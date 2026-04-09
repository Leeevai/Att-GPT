"""att_gpt: compact GPT training stack for educational use."""

from .config import ModelConfig, TrainConfig
from .model import GPTLanguageModel

__all__ = ["ModelConfig", "TrainConfig", "GPTLanguageModel"]
