# Contributing to att-gpt

Thanks for contributing. This project prioritizes readability, correctness, and educational clarity.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## Quality checklist

Before opening a PR:

1. Run smoke test:
   - `PYTHONPATH=src python -m pytest -q`
2. If training logic changed, run quick train smoke:
   - `PYTHONPATH=src python scripts/train.py --max-iters 2 --eval-interval 1 --eval-iters 1 --batch-size 4 --block-size 32 --n-embd 64 --n-head 4 --n-layer 2 --output-dir checkpoints/smoke`
3. Verify generation/benchmark scripts with the smoke checkpoint.
4. Update docs when behavior or interfaces change.

## Coding standards

- Keep modules focused and single-purpose.
- Prefer explicit names over compact one-liners for core logic.
- Avoid hidden side effects in utility functions.
- Add comments only for non-obvious algorithmic behavior.

## Documentation standards

Every meaningful implementation change should update one or more of:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TRAINING_RUNBOOK.md`
- `docs/REPOSITORY_GUIDE.md`

## Commit style

- Use concise imperative subject lines.
- Include rationale in commit body when changing architecture or training behavior.
- Keep unrelated changes in separate commits.
