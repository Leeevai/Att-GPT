# att-gpt

`att-gpt` is a compact, educational GPT-style language model trained on Shakespeare text.  
The project is intentionally small enough to read end-to-end while still implementing the full transformer decoder training loop used by modern autoregressive language models.

## Why this repository exists

- Teach transformer internals by implementing every core part in plain PyTorch.
- Provide a strong base model you can iterate on (bigger context windows, better optimizers, mixed precision, etc.).
- Keep architecture and implementation decisions explicit and documented.

## Project structure

```text
att-gpt/
├── data/raw/shakespeare.txt      # Training corpus (character-level)
├── docs/ARCHITECTURE.md          # Full architecture + design rationale
├── scripts/
│   ├── train.py                  # Main training CLI
│   ├── generate.py               # Text generation CLI
│   └── benchmark.py              # Inference benchmark CLI
├── src/att_gpt/
│   ├── config.py                 # Model/train dataclasses
│   ├── data.py                   # Corpus loading + batching
│   ├── device.py                 # Device selection helper
│   ├── model.py                  # Transformer decoder model
│   ├── tokenizer.py              # Character tokenizer
│   ├── trainer.py                # Training loop and checkpointing
│   └── benchmark/
│       ├── training.py           # Training benchmark utilities
│       └── inference.py          # Inference benchmark utilities
└── pyproject.toml
```

## Architecture (high level)

The model follows a decoder-only transformer stack:

1. Character tokens are embedded into dense vectors.
2. Learned positional embeddings are added.
3. Repeated transformer blocks perform:
   - Causal multi-head self-attention.
   - Position-wise feed-forward network.
   - Pre-layernorm + residual connections.
4. Final layer norm + linear language head produce token logits.
5. Training objective is next-token prediction via cross entropy.

Full details: `docs/ARCHITECTURE.md`.

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Train

```bash
PYTHONPATH=src python scripts/train.py --max-iters 2000 --eval-interval 200
```

Artifacts are written to `checkpoints/`:
- `model.pt`
- `tokenizer.json`

### 3) Generate text

```bash
PYTHONPATH=src python scripts/generate.py --tokens 400
```

### 4) Benchmark inference

```bash
PYTHONPATH=src python scripts/benchmark.py --tokens 300
```

## Implementation principles

- **Readable first:** minimal abstraction overhead.
- **Faithful transformer mechanics:** explicit causal mask, residuals, LN placement.
- **Measurable:** benchmark modules split from model/training logic.
- **Extensible:** dataclass configs and modular package structure.

## Next extensions

- Add mixed precision + gradient clipping.
- Add learning-rate warmup/cosine decay.
- Add BPE tokenization and larger corpora.
- Add checkpoint resumption and experiment tracking.
