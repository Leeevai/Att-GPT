from att_gpt.config import ModelConfig
from att_gpt.model import GPTLanguageModel


def test_forward_pass_shape() -> None:
    cfg = ModelConfig(vocab_size=65, block_size=32, n_embd=64, n_head=4, n_layer=2, dropout=0.0)
    m = GPTLanguageModel(cfg)

    import torch

    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = m(x, x)
    assert logits.shape == (4 * cfg.block_size, cfg.vocab_size)
    assert loss is not None
