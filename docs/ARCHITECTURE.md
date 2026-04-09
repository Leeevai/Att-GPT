# att-gpt Architecture and Design Notes

This document explains each major architectural decision and implementation detail in `att-gpt`.

## 1) Modeling choice: decoder-only transformer

`att-gpt` uses a **decoder-only** transformer because the training task is next-token prediction:

- Given tokens `t0..tN`, predict `t1..tN+1`.
- Self-attention is constrained with a **causal mask** so token `i` cannot see future tokens `> i`.
- This directly maps to text generation workloads and mirrors GPT-family pretraining.

Why this is ideal for a base educational model:

- Minimal architecture surface area.
- All modern LLM building blocks are present.
- Easy to extend while preserving conceptual clarity.

## 2) Tokenization: character-level

Tokenizer implementation is in `src/att_gpt/tokenizer.py`.

Why character-level for this base model:

- Eliminates tokenization complexity so focus remains on transformer mechanics.
- Shakespeare corpus has manageable alphabet size, leading to a small vocabulary.
- Easy debugging: each generated token maps to a visible character.

Trade-off:

- Longer effective sequences than subword tokenization.
- Lower semantic compression and slower convergence at scale.

## 3) Embedding strategy

In `src/att_gpt/model.py`:

- `token_embedding`: maps vocab IDs -> `n_embd` vectors.
- `position_embedding`: learned absolute positions `0..block_size-1`.
- Final input representation is `token_embedding + position_embedding`.

Rationale:

- Learned positional embeddings are simple, robust, and suitable for short educational contexts.
- Clear entry point for future upgrades (RoPE, ALiBi, relative positions).

## 4) Attention internals

`AttentionHead` computes:

1. Linear projections for `Q`, `K`, `V`.
2. Scaled dot-product attention scores.
3. Causal masking using lower-triangular matrix.
4. Softmax + dropout.
5. Weighted value aggregation.

`MultiHeadAttention` runs several heads in parallel, concatenates outputs, and projects back to `n_embd`.

Rationale:

- Multiple heads enable diverse subspace interactions.
- Projection layer fuses head-specific representations.

## 5) Transformer block design

Each block (`TransformerBlock`) uses:

- `x = x + attention(layernorm(x))`
- `x = x + feedforward(layernorm(x))`

This is a **pre-layernorm residual block**.

Why pre-norm:

- Better optimization stability as depth increases.
- More reliable gradient flow for educational experiments.

Feed-forward network:

- Expansion ratio 4x (`n_embd -> 4*n_embd -> n_embd`).
- ReLU nonlinearity.
- Dropout regularization.

## 6) Output head and objective

- Final layer norm improves representation conditioning.
- Linear `lm_head` projects hidden states to vocab logits.
- Cross-entropy is computed against next-token targets.

This is the canonical autoregressive language modeling objective.

## 7) Data pipeline

`src/att_gpt/data.py`:

- Loads raw corpus from `data/raw/shakespeare.txt`.
- Builds tokenizer from corpus symbols.
- Encodes full corpus to integer IDs.
- Splits contiguous stream into train/validation partitions.
- Samples random contiguous windows for each batch.

Windowed batching design:

- Inputs are sequences of length `block_size`.
- Targets are same sequences shifted by one token.
- Mirrors causal LM training exactly.

## 8) Benchmark architecture (split from training script)

The original monolithic benchmark logic is separated into:

- `src/att_gpt/benchmark/training.py`
  - Iteration timing
  - Throughput and parameter counts
  - Peak memory reporting (CUDA/MPS best effort)
- `src/att_gpt/benchmark/inference.py`
  - Warmup pass
  - Timed generation pass
  - Latency and tokens/sec

Reason for separation:

- Clear single-responsibility modules.
- Easier profiling experiments without touching model logic.
- Cleaner script interfaces.

## 9) Training orchestration

`src/att_gpt/trainer.py` manages:

- Device selection.
- Seed setup.
- Corpus construction and tokenizer persistence.
- Model/optimizer creation.
- Periodic train/validation loss estimation.
- Iteration benchmarking.
- Inference benchmarking and sample generation.
- Checkpoint serialization.

Saved artifacts:

- `checkpoints/model.pt`: state dict + config snapshot.
- `checkpoints/tokenizer.json`: reproducible encoding.

## 10) Why this is a strong base model

This codebase is intentionally simple but structurally complete:

- Implements all core GPT mechanisms.
- Keeps files small and role-focused.
- Includes objective metrics, checkpoints, and docs.

From here, you can evolve it into a research or production-like stack by adding:

- Mixed precision and gradient scaling.
- Learning-rate schedules and warmup.
- Better tokenization.
- Distributed training and logging.
