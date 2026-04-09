# Repository Guide

This guide documents ownership boundaries across modules so contributors can make changes confidently and safely.

## Top-level overview

- `scripts/`: thin CLI entrypoints for training, generation, and benchmarking.
- `src/att_gpt/`: core implementation package.
- `docs/`: architecture and operational documentation.
- `tests/`: automated checks.

## Core module responsibilities

### `src/att_gpt/config.py`

- Defines model and training dataclasses.
- Central place for hyperparameter defaults.
- Any new tunable should be represented here first.

### `src/att_gpt/device.py`

- Encapsulates runtime device selection (`cuda`, `mps`, `cpu`).
- Keeps platform logic out of training/model code.

### `src/att_gpt/tokenizer.py`

- Character-level vocabulary construction.
- Encoding/decoding primitives.
- Tokenizer serialization contract.

### `src/att_gpt/data.py`

- Corpus loading and split creation.
- Batch window sampling logic.
- Dataset-side modifications belong here, not in trainer.

### `src/att_gpt/model.py`

- Decoder-only transformer implementation.
- Attention head, multi-head attention, feed-forward, block, and LM head.
- Architectural experiments (activations, norm style, positional strategy) should be isolated to this module.

### `src/att_gpt/trainer.py`

- End-to-end training orchestration.
- Loss estimation policy.
- Checkpoint persistence contract.
- Model lifecycle and benchmark invocation.

### `src/att_gpt/benchmark/`

- `training.py`: per-iteration timing, throughput, memory metrics.
- `inference.py`: token generation speed and latency measurements.

## CLI entrypoint contract

- `scripts/train.py`: accepts user-facing training args and maps them to dataclasses.
- `scripts/generate.py`: loads checkpoint + tokenizer and runs generation.
- `scripts/benchmark.py`: loads checkpoint and reports inference metrics.

Root wrappers `train.py` and `benchmark.py` are backward compatibility shims.

## Data contract

- Canonical corpus path: `data/raw/shakespeare.txt`.
- Input format: UTF-8 plaintext.
- Tokenizer vocabulary is derived directly from loaded corpus.

## Checkpoint contract

Saved file: `checkpoints/model.pt` with:

- `state_dict`: PyTorch model parameters.
- `model_config`: serialized `ModelConfig`.
- `train_config`: serialized `TrainConfig` (with `output_dir` normalized to string).

## Extension map

Recommended sequence for contributors:

1. Add optimizer/scheduler options to `config.py`.
2. Wire behavior in `trainer.py`.
3. Expose flags via `scripts/train.py`.
4. Document updates in `docs/TRAINING_RUNBOOK.md`.
5. Add/adjust tests under `tests/`.
