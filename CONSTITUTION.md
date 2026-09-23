# skfin — Constitution

## Purpose

**Machine learning for portfolio management and trading**, built on scikit-learn conventions.

The library provides practitioners and students with:
1. A composable backtesting engine that integrates with sklearn pipelines
2. Portfolio construction estimators (mean-variance, Ridge, etc.)
3. Data loaders for standard financial datasets (Kenneth French, FOMC, etc.)
4. Notebooks that serve as both documentation and teaching material

## Scope

| In scope | Out of scope |
|----------|--------------|
| Estimators compatible with sklearn `fit`/`predict`/`transform` | Live trading / execution |
| Walk-forward backtesting with costs | Real-time data feeds |
| Portfolio risk & leverage management | Brokerage API integration |
| Text/NLP features for trading signals | Infrastructure / deployment |
| Datasets & caching utilities | |

## Design Principles

1. **sklearn-native** — Estimators follow the sklearn API (`BaseEstimator`, `TransformerMixin`). Users compose pipelines with `make_pipeline`.
2. **Notebook-first documentation** — Each concept is a self-contained notebook that runs end-to-end.
3. **Minimal dependencies** — Core depends on numpy, pandas, scikit-learn. Optional extras (NLP, LLM) are isolated.
4. **Separation of concerns** — Library code (`skfin/`) is independent of teaching material (`nbs/`) and student work.
5. **Readability above all** — This code is used for teaching. Clarity wins over cleverness. A student should be able to read a function and understand it without jumping to 5 other files.
6. **Maximum simplicity** — After each iteration, ask: "Could this have been coded in a simpler way?" If yes, simplify before moving on. Fewer abstractions, fewer indirections, fewer lines.
7. **Notebooks are the source of truth for `.py` files** — Library modules are written progressively using `%%writefile` magic inside notebooks, in notebook order. The `.py` files are a byproduct of running the notebooks. Exception: helper modules from appendix notebooks (90_, 91_, 92_) are always available as foundational utilities regardless of notebook order.

## Architecture

```
skfin/
├── skfin/                  # installable package
│   ├── backtesting/        # backtester engine, cost models
│   ├── estimators/         # sklearn-compatible estimators
│   ├── dataloaders/        # dataset fetchers & caching
│   ├── metrics/            # performance & risk metrics
│   ├── plot/               # visualization helpers
│   └── text/               # NLP/LLM feature extractors
├── nbs/                    # teaching notebooks
├── tests/                  # pytest suite
├── pyproject.toml          # single build config (PEP 621)
└── CONSTITUTION.md         # this file
```

## GSD Rules

- Ship working increments: each commit should leave the package importable and tests passing.
- No gold-plating: implement what exists today cleanly, don't add features.
- Decide fast: when in doubt, follow sklearn conventions.
- Delete aggressively: dead code is worse than missing code.

---

## History — the skfin2 refactor (completed)

This repository's package (`skfin/`) and teaching notebooks (`nbs/`) were rewritten from scratch under GSD discipline in a separate staging repo, `skfin2`, then merged back here once every notebook and the full test suite were green. `skfin2` no longer exists as a separate repo; this file and `STANDARDS.md` are its surviving output.

### Refactoring goals (all met)

1. **Clean package structure** — no leftover `old/`, `*_.py` files, or duplicated modules.
2. **Modern packaging** — single `pyproject.toml` with proper metadata, no `setup.py`/`requirements*.txt` proliferation.
3. **Testable** — unit tests for estimators and backtester, runnable with `pytest`.
4. **Clear boundaries** — library code has zero dependency on notebook paths or class/project folders.
5. **Documented API** — docstrings on public classes/functions following Google style (see `STANDARDS.md` §4).

### Step 1 — Notebook execution tests (the safety net used during the refactor)

Before touching any library code, a pytest-based safety net was built that executed every notebook end-to-end against the pre-refactor code, giving a pass/fail baseline so that refactoring which broke a notebook would be immediately visible. That suite now lives at `tests/test_notebooks.py` / `tests/conftest.py`, pointed at the local `nbs/` directory.

- **No skipping** — all notebooks run against the existing data cache in `nbs/data/`. Failures surface missing data or broken code.
- **Generous timeout** — some notebooks take 4+ minutes (e.g. `29_Ensemble`). Default timeout: 600s per notebook.
- **No network mocking** — if the cache is missing, the test fails (by design).

### Definition of done (met at merge time)

1. All notebooks migrated and passing their execution tests.
2. `pip install -e .` works from a clean environment.
3. `pytest` passes with zero failures.
4. No code from `old/`, no `setup.py`, no `requirements*.txt` remains.
5. Every public class/function has a one-line docstring.
6. `CONSTITUTION.md` and `STANDARDS.md` are up to date with any decisions made during migration.
