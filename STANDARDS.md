# skfin — Coding & Organisational Standards

## Glossary of Financial Abbreviations

| Symbol | Meaning |
|--------|---------|
| `h` | Holdings (portfolio weights) |
| `V` | Covariance matrix |
| `ret` | Returns |
| `pred` | Predicted returns |
| `pnl` | Profit & loss |
| `X` | Features matrix |
| `y` | Targets |
| `A` | Constraint matrix |

---

## 1. sklearn Conventions

### 1.1 Estimator API

All estimators follow the sklearn contract:

- Inherit from `BaseEstimator` (provides `get_params()`/`set_params()`)
- `fit(X, y=None)` → stores fitted attributes → returns `self`
- `predict(X)` → returns array or DataFrame
- `transform(X)` → alias for `predict()` where needed (enables pipeline use)
- `score(X, y)` → returns scalar performance metric

### 1.2 Fitted Attributes

Attributes set during `fit()` use a trailing underscore:

```python
self.V_        # fitted covariance matrix
self.h_        # fitted holdings
self.pnl_      # fitted P&L series
self.coef_     # fitted coefficients
self.estimators_  # list of fitted sub-estimators
```

### 1.3 Cloning and Cross-Validation

- Use `sklearn.base.clone()` to duplicate estimators for rolling CV
- Use `TimeSeriesSplit` for temporal validation (no future leakage)
- Estimator parameters must be fully determined by constructor arguments (for `clone()` to work)

### 1.4 Transform Alias

When a class needs both `predict()` and `transform()` (for pipeline compatibility), make the alias explicit in the class body:

```python
def transform(self, X):
    """Alias for predict (enables sklearn pipeline use)."""
    return self.predict(X)
```

---

## 2. Class Definition

### 2.1 `@dataclass` + `BaseEstimator`

Use `@dataclass` for all estimators and backtesters:

```python
@dataclass
class MeanVariance(BaseEstimator):
    transform_V: Callable = field(default=lambda x: np.cov(x.T))
    A: np.ndarray | str | None = "cash-neutral"
    risk_target: float = 1.0
```

Benefits: no boilerplate `__init__`, type-annotated fields, compatible with `get_params()`.

### 2.2 Method Chaining

Mutating methods return `self`. Access results via fitted attributes:

```python
bt = Backtester(estimator=Ridge()).compute_holdings(X, y).compute_pnl(ret)
result = bt.pnl_  # access the result
```

Never return data from a mutating method. This keeps the API predictable:
- If a method modifies `self` → returns `self`
- If a method is a pure function → returns data (and lives as a module-level function, not a method)

---

## 3. Type Annotations

Use Python 3.10+ union syntax. No `from typing import Optional, Union`:

```python
# Yes:
def foo(x: np.ndarray | str | None = None) -> dict:

# No:
from typing import Optional, Union
def foo(x: Optional[Union[np.ndarray, str]] = None) -> Dict:
```

Retain `from typing import Callable` and `from dataclasses import dataclass, field` as needed.

---

## 4. Docstrings

### When to Write Them

| Element | Docstring? | Content |
|---------|-----------|---------|
| Public class | Yes | One sentence: what it does |
| Public function with non-obvious behavior | Yes | Args/Returns, Google style |
| Trivial methods (`fit(X, y)`, `predict(X)`) | No | Signature is self-documenting |
| Private helpers (`_foo()`) | No | Unless the logic is subtle |

### Style

Google style, concise:

```python
def compute_batch_holdings(pred, V, A=None, risk_target=None):
    """Compute Markowitz portfolio holdings from predictions and covariance.

    Args:
        pred: Return predictions, shape (N, K).
        V: Covariance matrix, shape (N, N).
        A: Constraint matrix or "cash-neutral". None for unconstrained.
        risk_target: Target portfolio volatility. None disables scaling.

    Returns:
        Holdings array, shape (N, K).
    """
```

### What NOT to Document

- Do not restate the function name ("Compute holdings" on `compute_holdings()`)
- Do not document internal mechanics that a student can read from the code
- Do not add docstrings to unchanged code during migration

---

## 5. Error Handling

### At Public API Boundaries

Validate inputs with `raise ValueError(...)`:

```python
def fit(self, X, y=None):
    if X.shape[0] < self.max_train_size:
        raise ValueError(f"X has {X.shape[0]} rows, need at least {self.max_train_size}")
```

### Internally

Trust the caller. No defensive coding inside module internals. Let numpy/pandas raise natural errors — they are clear enough for debugging.

### Never Use `assert` for Validation

Assertions are stripped with `python -O`. Use explicit `if`/`raise` at boundaries.

---

## 6. Boolean Operators

Use `and`/`or` for control flow. Reserve `&`/`|` for element-wise pandas/numpy operations:

```python
# Yes:
if lambda_ is None and vol_liquidity_factor is not None:

# No:
if (lambda_ is None) & (vol_liquidity_factor is not None):
```

---

## 7. Imports

### Ordering (PEP 8)

1. Standard library
2. Third-party packages
3. Local imports

### Rules

- No `from module import *`
- Explicit imports only
- Standard aliases: `import numpy as np`, `import pandas as pd`, `from matplotlib import pyplot as plt`

---

## 8. Logging

Library modules never configure the logging system:

```python
# Yes — in every module:
import logging
logger = logging.getLogger(__name__)

# No — never in library code:
logging.basicConfig(...)
```

Notebooks and applications configure handlers. The library only emits log messages.

---

## 9. Global State

Library modules must not mutate global state on import:

```python
# No — not at module level:
plt.style.use("seaborn-whitegrid")
pd.options.display.max_colwidth = None

# Yes — inside functions or notebooks:
with plt.style.context("seaborn-whitegrid"):
    line(df)
```

Notebooks may set global style at the top. Library code never does.

---

## 10. Module Organisation

### Size

Soft cap: ~200 lines per module. If a module grows beyond that, split by responsibility.

### File Naming

- Snake case: `mean_variance.py`, `backtesting.py`
- No trailing underscore hacks (`datasets_.py` → `datasets.py`)
- Subdirectories for cohesive groups: `estimators/`, `dataloaders/`, `backtesting/`

### Package Layout (target)

```
skfin/
├── backtesting/        # backtester engine, cost models
├── estimators/         # sklearn-compatible estimators
├── dataloaders/        # dataset fetchers & caching
├── metrics/            # performance & risk metrics
├── plot/               # visualization helpers
└── text/               # NLP/LLM feature extractors
```

---

## 11. Notebooks as Source of Truth

### Markdown Cell Formatting

- Section/subsection titles (`#`, `##`, `###`) must be in their own cell — never combined with body text.
- Titles must not contain colons. Use a separate body cell for the description.
- `# hide` comments in code cells are intentional (nbdev convention for hiding cells in exported docs). Do not remove or flag them.

### `%%writefile` Convention

Library `.py` files are generated by running notebooks. The notebooks are the authoritative source.

### Independence Rule

Each notebook's `%%writefile` cells must be self-contained. Running notebook N must never require running notebook M first.

**Exception:** Helper notebooks (90_, 91_, 92_) are foundational utilities — always available and assumed to have been run first.

If a circular dependency exists between notebooks, it signals a design problem to fix.

### Notebook Order = Dependency Order

Migration order (helpers first → domain modules → introduction last) must work without backtracking. The `/migrate-notebook` skill enforces this order and the independence rule during migration.

1. `90_Helper_functions`, `91_Helper_text_visualisation`, `92_Project_template`
2. `05_Data`
3. `10_Mean_variance_estimators`
4. `11_Backtesting`
5. Remaining notebooks in numerical order
6. `01_Introduction` (last — imports everything)

---

## 12. Data & Caching

### Filesystem Cache

- Cache lives in `nbs/data/` (relative to notebook working directory)
- Format auto-detected: parquet preferred, csv/pickle as fallback
- `force_reload: bool = False` parameter on all loaders to bypass cache
- No network mocking — if cache is missing, the code fails (by design)

### LLM Cache

Generic disk cache decorator for LLM calls:
- Keyed by `(model, prompt)` hash
- Stored in `data/` alongside other cached datasets
- Enables offline notebook execution without API keys

---

## 13. Parallelisation

- `joblib.Parallel` + `delayed` for CPU-bound rolling CV (sklearn standard)
- `concurrent.futures.ThreadPoolExecutor` for I/O-bound calls (LLM, web)
- `n_jobs` parameter on backtesters for user control

---

## 14. Dependencies

### Core (required)

numpy, pandas, scikit-learn, matplotlib, joblib

### Optional Extras

Isolated behind import guards or optional dependency groups:

| Group | Packages |
|-------|----------|
| `nlp` | sentence-transformers, beautifulsoup4 |
| `llm` | openai, tenacity |
| `boost` | lightgbm |
| `optim` | cvxpy |

A missing optional dependency raises `ImportError` with a clear message at call time, not at import time.

---

## 15. Testing

- `pytest` as the test runner
- Notebook execution tests: parametrize over all `.ipynb`, execute with `nbconvert.ExecutePreprocessor`
- Unit tests for estimators: verify `fit`/`predict`/`clone` contract
- Timeout: 600s per notebook (some run 4+ minutes)
- No skipping — all notebooks must pass. Failures surface real problems.

### Contract Tests

Contract tests verify **behavioral invariants** — properties that must always hold regardless of implementation details. They use small synthetic data, run in milliseconds, and catch regressions without notebook overhead.

#### Which modules get contracts

| Module | Contract? | Reason |
|--------|-----------|--------|
| Estimators (MeanVariance, TimingMeanVariance) | **Yes** | Mathematical invariants from financial theory |
| Backtester, BacktesterWithCost | **Yes** | Temporal guarantees (no leakage), shape contracts |
| Dataloaders | No | I/O, no behavioral invariants |
| Plot | No | Visual output, not testable invariants |
| Text/LLM | No | Output depends on external models |

#### Structure

Contract tests live in `tests/contracts/`, one file per module:

```
tests/contracts/
├── test_mean_variance.py
├── test_mean_variance_with_cost.py
├── test_backtester.py
└── test_backtester_with_cost.py
```

#### What to test

**MeanVariance:**
- Cash-neutral constraint: `h.sum(axis=1) ≈ 0` when `A="cash-neutral"`
- Risk scaling: `sqrt(h.T @ V @ h) ≈ risk_target` when `risk_target` is set
- Unconstrained case: `h = V⁻¹ @ pred` when `A=None`
- Shape: `predict(X)` returns `(K, N)` — K prediction periods, N assets
- Fit stores covariance: `self.V_` is `(N, N)` and symmetric after `fit()`
- sklearn clone: `clone(estimator).get_params()` round-trips correctly

**Backtester:**
- No future leakage: `h_` at time t depends only on data ≤ t-1
- PnL computation: `pnl_ = h.shift(pred_lag) * ret`, summed across assets
- Index alignment: `h_.index` and `pnl_.index` are subsets of `X.index`
- Date filtering: `pnl_` is clipped to `[start_date, end_date]`
- Chainable: `compute_holdings()` and `compute_pnl()` return `self`
- Holdings shape: `h_` has same columns as `y`

**MeanVarianceWithCost:**
- Degenerates to base: without cost parameters, produces same result as `MeanVariance`
- Cost reduces turnover: with `vol_liquidity_factor > 0`, consecutive holdings are closer together
- Past holdings influence: with `past_h`, solution is pulled toward previous position

**BacktesterWithCost:**
- PnL components: when `return_pnl_component=True`, returns dict with keys `"gross"`, `"net = gross - impact cost"`, `"impact cost"`
- Impact cost is non-positive: costs always reduce returns
- Degenerates to base: without cost, matches `Backtester` output

#### Design rules for contract tests

- Use deterministic synthetic data (fixed seed, small matrices: 5 assets, 50 time steps)
- Each test verifies exactly one invariant
- No dependency on data files or network
- Must run in < 1 second total for all contract tests
- Contract tests are created as part of the `/migrate-notebook` workflow for applicable modules

---

## 16. Packaging

- Single `pyproject.toml` (PEP 621) — no `setup.py`, no `requirements*.txt`
- Package metadata, dependencies, optional extras, and tool config all in one file
- Editable install: `pip install -e .` for development

---

## 17. GSD Rules

- Ship working increments: each commit leaves the package importable and tests passing
- No gold-plating: implement what exists today cleanly, don't add features
- Decide fast: when in doubt, follow sklearn conventions
- Delete aggressively: dead code is worse than missing code
- Maximum simplicity: after each iteration, ask "could this be simpler?" — if yes, simplify before moving on
