# att-gpt

`att-gpt` is a compact, professional GPT-style baseline trained on Shakespeare writings.
It is intentionally designed to be small enough to understand line-by-line while still implementing a complete decoder-only transformer training stack.

## Project goals

- Implement a complete autoregressive transformer in clean, explicit PyTorch.
- Explain every major design and implementation decision in one place.
- Provide reproducible training, generation, and benchmarking workflows.
- Keep the codebase extensible for iterative research and engineering upgrades.

## Repository structure

```text
att-gpt/
├── data/raw/shakespeare.txt
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
├── benchmark.py                 # backward-compatible wrapper
├── train.py                     # backward-compatible wrapper
└── pyproject.toml
```

## Architecture and rationale

### 1) Model family: decoder-only transformer

`att-gpt` uses a decoder-only transformer because the training objective is next-token prediction.
Each token predicts the following token in sequence, and causal masking prevents leakage from future positions.

Why this architecture:

- Matches the standard GPT autoregressive objective.
- Contains all core modern LLM building blocks.
- Keeps conceptual complexity manageable for learning and extension.

### 2) Tokenization: character-level

Tokenizer logic is implemented in `src/att_gpt/tokenizer.py`.

Why character-level for this baseline:

- Removes subword-tokenizer complexity so focus remains on transformer mechanics.
- Shakespeare corpus has a manageable character vocabulary.
- Makes debugging transparent because each token maps directly to a visible character.

Trade-offs:

- Longer sequences than subword approaches.
- Lower semantic compression at larger scale.

### 3) Embeddings

In `src/att_gpt/model.py`, input representation is:

- Token embedding (`vocab_size -> n_embd`)
- Plus learned absolute positional embedding (`block_size -> n_embd`)

This keeps the implementation direct and provides a clean upgrade path for RoPE/relative methods later.

### 4) Attention internals

`AttentionHead` performs:

1. Linear projections for query/key/value.
2. Scaled dot-product score computation.
3. Causal mask application.
4. Softmax normalization and dropout.
5. Weighted aggregation over values.

`MultiHeadAttention` runs heads in parallel, concatenates outputs, and applies an output projection.

### 5) Transformer block design

Each block uses pre-layernorm residual structure:

- `x = x + attention(layernorm(x))`
- `x = x + feedforward(layernorm(x))`

Why pre-norm:

- Better optimization stability for deeper stacks.
- More robust gradient flow in practice.

Feed-forward network is the standard 4x expansion MLP:

- `n_embd -> 4*n_embd -> n_embd`
- ReLU activation
- Dropout regularization

### 6) Output head and training objective

- Final layer norm before output projection.
- Linear language-model head projects to vocabulary logits.
- Cross-entropy loss against shifted targets (next-token objective).

### 7) Data pipeline

Data loading in `src/att_gpt/data.py`:

- Load UTF-8 text from `data/raw/shakespeare.txt`.
- Build vocabulary from unique characters.
- Encode corpus into integer token IDs.
- Split stream into train/validation partitions.
- Sample random contiguous windows of length `block_size`.
- Build shifted targets (`x[t+1]`) for causal LM training.

### 8) Benchmark design

Benchmark logic is intentionally split from training code:

- `src/att_gpt/benchmark/training.py`
  - Iteration timing
  - Throughput estimates
  - Parameter count and memory reporting
- `src/att_gpt/benchmark/inference.py`
  - Warmup pass
  - Timed generation
  - Tokens/sec and latency metrics

This separation improves readability and makes profiling experiments safer.

### 9) Training orchestration

`src/att_gpt/trainer.py` owns:

- Device detection and seed setup
- Corpus and tokenizer lifecycle
- Model and optimizer creation
- Periodic train/validation loss estimation
- Training/inference benchmark calls
- Checkpoint persistence

Saved artifacts:

- `checkpoints/model.pt`
- `checkpoints/tokenizer.json`

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Baseline training run

```bash
PYTHONPATH=src python scripts/train.py   --data-path data/raw/shakespeare.txt   --batch-size 64   --block-size 256   --max-iters 5000   --eval-interval 500   --eval-iters 200   --learning-rate 3e-4   --n-embd 384   --n-head 6   --n-layer 6   --dropout 0.2   --output-dir checkpoints
```

### 3) Generate text

```bash
PYTHONPATH=src python scripts/generate.py   --checkpoint checkpoints/model.pt   --tokenizer checkpoints/tokenizer.json   --tokens 400
```

### 4) Inference benchmark

```bash
PYTHONPATH=src python scripts/benchmark.py   --checkpoint checkpoints/model.pt   --tokens 300
```

## Training runbook

### Smoke validation run

Use this before major refactors or PRs:

```bash
PYTHONPATH=src python scripts/train.py   --max-iters 2   --eval-interval 1   --eval-iters 1   --batch-size 4   --block-size 32   --n-embd 64   --n-head 4   --n-layer 2   --output-dir checkpoints/smoke
```

Then verify:

```bash
PYTHONPATH=src python scripts/generate.py   --checkpoint checkpoints/smoke/model.pt   --tokenizer checkpoints/smoke/tokenizer.json   --tokens 40
```

```bash
PYTHONPATH=src python scripts/benchmark.py   --checkpoint checkpoints/smoke/model.pt   --tokens 40
```

### Interpreting metrics

- `train loss`: fit quality on sampled training windows.
- `val loss`: generalization signal on held-out split.
- `avg iter (ms)`: mean per-step training time.
- `throughput tok/s`: training token processing rate.

### Tuning guidance

- Increase `n_embd` and `n_layer` for capacity, with higher runtime cost.
- Increase `block_size` for context length, with higher memory cost.
- Start with `learning-rate=3e-4`; lower it if loss oscillates.
- Keep `dropout` around `0.1-0.2` for this corpus scale.

### Common issues

#### `ModuleNotFoundError: No module named 'torch'`

```bash
pip install -e .
```

#### Missing checkpoint files

Ensure training completed successfully and check:

- `checkpoints/model.pt`
- `checkpoints/tokenizer.json`

#### Device differences

Runtime priority is:

1. `cuda`
2. `mps`
3. `cpu`

Only compare throughput and latency on the same device class.

## Module ownership guide

### `src/att_gpt/config.py`

- Defines all model/training defaults.
- Any new tuning knob should be added here first.

### `src/att_gpt/device.py`

- Centralizes hardware selection logic.
- Keeps device branching out of model/trainer internals.

### `src/att_gpt/tokenizer.py`

- Vocabulary construction.
- Encoding/decoding behavior.
- Tokenizer serialization contract.

### `src/att_gpt/data.py`

- Corpus ingestion and split strategy.
- Batch sampling logic.

### `src/att_gpt/model.py`

- Core transformer architecture.
- Best place for attention/norm/activation experiments.

### `src/att_gpt/trainer.py`

- Training loop, evaluation, benchmarking orchestration.
- Checkpoint write contract.

### `src/att_gpt/benchmark/`

- `training.py`: training-time profiling metrics.
- `inference.py`: generation latency/throughput metrics.

### `scripts/*.py`

- Stable user-facing CLI interfaces.
- Thin wrappers over package internals.

## Contribution standards

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

### Quality checklist

Before opening a PR:

1. `PYTHONPATH=src python -m pytest -q`
2. If training code changed, run the smoke training workflow.
3. Validate `scripts/generate.py` and `scripts/benchmark.py` with smoke artifacts.
4. Update this README whenever interfaces or behavior change.

### Coding standards

- Keep modules focused and single responsibility.
- Prefer explicit names over compact but opaque shortcuts.
- Avoid hidden side effects in helpers.
- Add comments only where behavior is non-obvious.

### Commit standards

- Use concise imperative commit subjects.
- Include rationale in body for architecture or training changes.
- Separate unrelated changes into distinct commits.

## Roadmap (Issue 0001)

Primary next milestone:

- Gradient clipping for training stability.
- Warmup + cosine LR schedule.
- Optional resume-from-checkpoint training.
- Training/validation loss plotting utility.

Acceptance criteria:

- Reproducible runs with fixed seed.
- Persisted loss curves.
- Documented throughput impact.
- Improved sample coherence at equal training budget.

## License and usage

This project is intended as an educational base model. Adapt, extend, and benchmark responsibly.
