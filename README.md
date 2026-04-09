# att-gpt

`att-gpt` is a compact, professional, and fully documented GPT-style baseline trained on Shakespeare writings.
It is designed for engineers who want to deeply understand transformer internals and evolve a clean base model into more advanced experiments.

## Project goals

- Implement a complete decoder-only transformer in clear PyTorch modules.
- Keep every architectural choice explicit and documented.
- Provide reproducible training, generation, and benchmarking workflows.
- Offer a robust foundation for iterative model development.

## Documentation index

- `docs/ARCHITECTURE.md` - full transformer architecture and design rationale.
- `docs/REPOSITORY_GUIDE.md` - repository layout, ownership, and extension map.
- `docs/TRAINING_RUNBOOK.md` - training workflows, hyperparameter guidance, and troubleshooting.
- `CONTRIBUTING.md` - contribution and quality standards.

## Repository structure

```text
att-gpt/
├── data/raw/shakespeare.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── REPOSITORY_GUIDE.md
│   ├── TRAINING_RUNBOOK.md
│   └── issues/
├── scripts/
│   ├── train.py
│   ├── generate.py
│   └── benchmark.py
├── src/att_gpt/
│   ├── benchmark/
│   ├── config.py
│   ├── data.py
│   ├── device.py
│   ├── model.py
│   ├── tokenizer.py
│   └── trainer.py
├── tests/
└── pyproject.toml
```

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Train a baseline checkpoint

```bash
PYTHONPATH=src python scripts/train.py --max-iters 2000 --eval-interval 200
```

Artifacts are saved under `checkpoints/`:
- `model.pt`
- `tokenizer.json`

### 3) Generate text from the checkpoint

```bash
PYTHONPATH=src python scripts/generate.py --tokens 400
```

### 4) Measure inference throughput

```bash
PYTHONPATH=src python scripts/benchmark.py --tokens 300
```

## Reproducibility and quality controls

- Deterministic seed in `TrainConfig`.
- Isolated benchmark modules for transparent performance measurement.
- Smoke test at `tests/test_smoke.py`.
- Backward-compatible root entrypoints (`train.py`, `benchmark.py`) preserved.

## Why this baseline is practical

- Small enough to understand fully, yet complete enough for serious iteration.
- Explicit causal masking and pre-layernorm residual design.
- Clean separation between model, data, training orchestration, and benchmarks.

## Immediate roadmap

- Gradient clipping and scheduler support.
- Resume-from-checkpoint training.
- Richer experiment logging and curves.
- Tokenization upgrades for larger corpora.
