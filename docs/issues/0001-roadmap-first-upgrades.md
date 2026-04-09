# Issue 0001 - First Upgrade Roadmap

## Title
Upgrade `att-gpt` baseline with training stability and quality improvements.

## Context
The current baseline is intentionally minimal and educational. The next milestone should improve optimization stability and generation quality while preserving readability.

## Scope

- Add gradient clipping to reduce occasional unstable updates.
- Add cosine decay learning-rate schedule with warmup.
- Add optional checkpoint resume support (`--resume-from`).
- Add train/validation loss plotting script.

## Acceptance Criteria

- [ ] Training remains reproducible with fixed seed.
- [ ] Loss curves are logged and saved to disk.
- [ ] Throughput impact is documented.
- [ ] Generated samples show improved coherence after equal training budget.
