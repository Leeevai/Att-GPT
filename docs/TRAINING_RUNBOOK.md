# Training Runbook

This runbook defines how to run, evaluate, and troubleshoot `att-gpt` training experiments.

## Baseline run

```bash
PYTHONPATH=src python scripts/train.py \
  --data-path data/raw/shakespeare.txt \
  --batch-size 64 \
  --block-size 256 \
  --max-iters 5000 \
  --eval-interval 500 \
  --eval-iters 200 \
  --learning-rate 3e-4 \
  --n-embd 384 \
  --n-head 6 \
  --n-layer 6 \
  --dropout 0.2 \
  --output-dir checkpoints
```

## Minimal smoke run

Use this before opening PRs or when validating refactors:

```bash
PYTHONPATH=src python scripts/train.py \
  --max-iters 2 \
  --eval-interval 1 \
  --eval-iters 1 \
  --batch-size 4 \
  --block-size 32 \
  --n-embd 64 \
  --n-head 4 \
  --n-layer 2 \
  --output-dir checkpoints/smoke
```

Then verify:

```bash
PYTHONPATH=src python scripts/generate.py \
  --checkpoint checkpoints/smoke/model.pt \
  --tokenizer checkpoints/smoke/tokenizer.json \
  --tokens 40
```

```bash
PYTHONPATH=src python scripts/benchmark.py \
  --checkpoint checkpoints/smoke/model.pt \
  --tokens 40
```

## Interpreting training output

- `train loss`: short-horizon fit quality on sampled train windows.
- `val loss`: generalization proxy on held-out contiguous slice.
- `avg iter (ms)`: mean step time from benchmark tracker.
- `throughput tok/s`: sampled tokens processed per second.

## Hyperparameter tuning guidance

- Increase `n_embd` and `n_layer` for quality; expect slower iteration speed.
- Increase `block_size` for longer context modeling; memory usage rises.
- Start with `learning-rate=3e-4`; reduce if loss oscillates.
- Keep `dropout` near `0.1-0.2` for this data scale.

## Common issues

### `ModuleNotFoundError: No module named 'torch'`

Install dependencies:

```bash
pip install -e .
```

### Missing checkpoint files

Confirm output path and successful train completion:

- `checkpoints/model.pt`
- `checkpoints/tokenizer.json`

### Device differences

Runtime chooses:

1. `cuda` if available
2. `mps` if available
3. `cpu` fallback

Performance numbers should be compared only on the same device type.
