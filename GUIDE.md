# SBBTS Data Guide

This guide covers what types of data SBBTS can model, the theoretical conditions
for applicability, how to use multi-feature input, and generic (non-financial)
workflows.

---

## Theoretical acceptance conditions

SBBTS models a joint distribution `µ ∈ P((R^d)^{n+1})` — any multivariate
time series in R^d. Two mathematical conditions must hold for the algorithm to
be well-defined (Theorem 3.2, Assumption 3.1 in the paper):

1. **β·Δt > 1** on every time interval (Δt = T / (n_time_points − 1)).
   Use `SBBTS.suggest_beta(n_time_points)` to satisfy this automatically.

2. **Finite KL divergence** at each step: the conditional distribution
   `µ_{i+1 | 0:i}` must be absolutely continuous with respect to the
   Brownian motion increment N(0, Δt·I_d) and satisfy
   `KL(µ_{i+1|0:i} | N_{Δt}) < ∞`.

   In practice this means:
   - The data must be **stationary** (or made stationary by preprocessing).
   - The per-step distribution must have a **smooth density** — no point masses,
     no perfectly discrete support.
   - **Heavy tails** are fine as long as the distribution has finite second moments.

---

## Accepted data types

| Data type | Support | Key condition | Recommended preprocessing |
|---|---|---|---|
| **Log-returns** | ✅ Validated (paper) | Stationary, smooth density | `normalize_input=True` (default) |
| **Simple returns** | ✅ OK | Same as log-returns | `normalize_input=True` |
| **Realized variance / volatility** | ✅ OK | Stationary, positive values | `np.log(rv)` if log-normal; then normalize |
| **PCA factors** | ✅ OK | Orthogonal, zero-mean | Already normalized; keep as-is |
| **Macro factors** (rates, spreads, FX) | ⚠️ Conditional | Must be **stationary** | Difference I(1) series first |
| **Trading volumes** | ⚠️ Conditional | Heavy right tail — log-transform advised | `np.log(vol + 1)` then normalize |
| **Raw prices** | ❌ Not recommended | Non-stationary (I(1) or I(2)) | Differentiate to returns first |
| **Non-financial continuous data** | ✅ In theory | Stationary, continuous density | Verify stationarity; normalize |
| **Discrete / categorical data** | ❌ Not applicable | KL divergence = +∞ vs N(0, Δt) | Not applicable |

**Rule of thumb**: if the series has a smooth, stationary marginal distribution with
finite variance, SBBTS will work. If it has structural breaks, unit roots, or a
discrete support, preprocess first.

---

## Multi-feature input (d > 1)

SBBTS natively accepts any number of simultaneous features. The input tensor
shape is `(N_windows, T, d)` where `d` is the number of features.

All internal components (transformer encoder, score network, transport map,
normalization) scale with `d` — there are no hardcoded univariate assumptions.

### d = 2: price log-return + realized variance

```python
import numpy as np
from sbbts import SBBTS

# Build rolling windows for two features
log_ret    = ...    # (N_days,) log-returns
realized_v = ...    # (N_days,) daily realized variance

T = 252
windows_r  = np.lib.stride_tricks.sliding_window_view(log_ret, T)      # (N, T)
windows_rv = np.lib.stride_tricks.sliding_window_view(realized_v, T)   # (N, T)

# Stack into (N, T, 2)
X_train = np.stack([windows_r, windows_rv], axis=-1).astype(np.float32)

beta = SBBTS.suggest_beta(n_time_points=T)
model = SBBTS(beta=beta, n_steps=5)
model.fit(X_train, feature_names=["log_return", "realized_vol"])

X_synth = model.sample(n=500)    # (500, T, 2)
```

### d = 5: multi-asset portfolio

```python
returns_matrix = ...    # (N_days, 5) returns for 5 assets

T = 252
windows = np.stack(
    [np.lib.stride_tricks.sliding_window_view(returns_matrix[:, i], T) for i in range(5)],
    axis=-1,
).astype(np.float32)   # (N, T, 5)

model = SBBTS(beta=SBBTS.suggest_beta(T), n_steps=5)
model.fit(windows, feature_names=["AAPL", "MSFT", "AMZN", "GOOGL", "META"])

X_synth = model.sample(n=500)    # (500, T, 5) — preserves cross-asset correlations
```

### d > 20: reduce first with PCA

For large d, reduce to a lower-dimensional latent space before training.
The model learns on the reduced representation and projects back to the
original space at sampling time.

```python
from sbbts import SBBTS, PCAKMeansReducer

reducer = PCAKMeansReducer(n_components=16, n_clusters=3)
X_reduced = reducer.fit_transform(X_high_dim)   # (N, T, 16)

model = SBBTS(beta=SBBTS.suggest_beta(T), dim_reducer=reducer)
model.fit(X_high_dim)               # handles reduction internally
X_synth = model.sample(n=500)      # returned in original (high-d) space
```

---

## Generic workflow (any data domain)

Use `diagnose_generic` and `compute_generic_metrics` when your data is not
financial returns. These functions make no assumptions about the data domain.

```python
import numpy as np
from sbbts import SBBTS, diagnose_generic, compute_generic_metrics

# Example: macro time series (interest rate + credit spread + FX)
X_train = ...    # (N, T, 3) — any stationary multivariate series

beta = SBBTS.suggest_beta(n_time_points=X_train.shape[1])
model = SBBTS(beta=beta, n_steps=5)
model.fit(X_train, feature_names=["rate", "spread", "fx"])

X_synth = model.sample(n=500)

# Generic diagnosis — no VaR, no Sharpe, no leverage effect
fig = model.diagnose_generic(X_train)

# Or as a standalone function
fig = diagnose_generic(X_train, X_synth, feature_names=["rate", "spread", "fx"])
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")

# Domain-agnostic metrics
metrics = compute_generic_metrics(X_train, X_synth)
print(f"Std ratio per feature:    {metrics['std_ratio']}")
print(f"Mean std ratio:           {metrics['std_ratio_mean']:.3f}  (target 1.0)")
print(f"Cross-feature corr error: {metrics['corr_frob_error']:.3f}  (target 0.0)")
print(f"KS statistic per feature: {metrics['ks_statistic']}")
```

### Generic plot functions

```python
from sbbts import (
    plot_feature_paths,       # one trajectory panel per feature
    plot_feature_marginals,   # one histogram per feature
    plot_feature_acf,         # one ACF panel per feature
    plot_feature_stats,       # heatmap of synth/real stat ratios
    diagnose_generic,         # composite of all the above
)

# Individual panels (useful for custom layouts)
plot_feature_paths(X_train, X_synth, feature_names=["rate", "spread", "fx"])
plot_feature_marginals(X_train, X_synth, feature_names=["rate", "spread", "fx"])
plot_feature_acf(X_train, X_synth, feature_names=["rate", "spread", "fx"])
plot_feature_stats(X_train, X_synth, feature_names=["rate", "spread", "fx"])
```

---

## Financial data (returns) — original use case

For financial log-returns, the specialized functions in `sbbts.utils.visualization`
give richer diagnostics (VaR, Sharpe, leverage effect, rolling volatility, TSTR):

```python
from sbbts import diagnose, full_diagnose, compute_metrics

fig = model.diagnose(X_real)          # 6-panel: paths, dist, ACF, corr, risk, lag-corr
fig = full_diagnose(X_real, X_synth)  # 12-panel: + cluster, leverage, rolling vol

metrics = compute_metrics(X_real, X_synth)   # financial stylized facts
print(f"Ann std ratio: {metrics['ann_std_ratio']:.3f}")
print(f"ACF|r| ratio:  {metrics['acf_abs_sum_ratio']:.3f}")
```

The `feature_names` parameter propagates automatically when set via `fit()`:

```python
model.fit(X_returns, feature_names=["SPX"])
fig = model.diagnose_generic(X_real)    # "SPX" appears on axes automatically
```

---

## Evaluating augmentation quality

`evaluate_augmentation()` runs a TSTR (Train-on-Synthetic, Test-on-Real) benchmark
with zero configuration — no custom function needed:

```python
result = model.evaluate_augmentation(X_real)

# result["tstr"] contains:
print(f"TSTR ratio  : {result['tstr']['ratio']:.4f}")   # target < 1.05
print(f"TRTR MSE    : {result['tstr']['trtr_mse']:.8f}")
print(f"TSTR MSE    : {result['tstr']['tstr_mse']:.8f}")
```

You can also pass a custom downstream model function for backward compatibility:

```python
from sklearn.linear_model import Ridge

def my_downstream(X_synth, X_real_test):
    # train on synthetic, score on real
    ...
    return score

result = model.evaluate_augmentation(X_real, downstream_model_fn=my_downstream)
```

---

## Conditional generation (fan charts)

`sample_conditional()` generates plausible continuations of an observed prefix:

```python
T_prefix = 50
X_prefix = X_train[-1, :T_prefix, :]       # (T_prefix, d) — observed history
X_fan    = model.sample_conditional(X_prefix, n=200)  # (200, T, d)

# Batch prefix: same prefix for all n paths
# X_prefix can also be (B, T_prefix, d) for B different prefixes
```

Applications: conditional stress testing, scenario fan charts, rolling forecast
simulation. The prefix is matched via the learned transport map — not a naive
zero-order hold.

---

## Reproducibility and scheduler

```python
# Fully reproducible run
model = SBBTS(beta=beta, n_steps=5, seed=42, lr_scheduler="cosine")
model.fit(X_train, checkpoint_dir="runs/exp1/")

# lr_scheduler="cosine": learning rate decays from lr to lr/100 each outer step
# lr_scheduler="none":   constant learning rate
# seed: fixes torch + numpy + CUDA RNG at the start of fit()
```

---

## Automatic data checks

`fit()` runs a set of sanity checks before training and reports problems as
`UserWarning` or raises `ValueError` for hard failures. The checks run by default
(`check_input=True`); pass `check_input=False` to skip them.

| Check | Trigger | Level | What to do |
|---|---|---|---|
| Near-constant feature | std < 1e-6 | **ValueError** | Remove the feature |
| Non-stationary (ADF test) | ADF p > 0.05 | Warning | Differentiate the series (e.g. returns instead of prices) |
| Near-discrete values | n_unique / n_total < 1% | Warning | Smooth the data or use a different representation |
| Extreme kurtosis | excess kurtosis > 100 | Warning | Winsorize or log-transform |
| Scale mismatch between features | max_std / min_std > 1000 | Warning | Verify features are meaningful; `normalize_input=True` handles scale |

The ADF test requires `statsmodels` (`pip install statsmodels`). Without it, SBBTS
falls back to a simpler linear-trend test.

Example output on non-stationary input:

```
UserWarning: [SBBTS data check] Feature 'price' may be non-stationary (ADF p=0.98 > 0.05).
Non-stationary series violate the KL condition (Assumption 3.1).
Consider differencing (e.g. log-returns instead of prices).
```

To suppress all checks (e.g. in a training loop where you already validated the data):

```python
model.fit(X_train, check_input=False)
```

---

## Practical checklist before training

- [ ] **Stationarity**: each feature should not trend over time. If prices → differentiate. If rates are I(1) → first-difference.
- [ ] **Scale**: use `normalize_input=True` (default). Especially important if features have very different scales.
- [ ] **Window count**: N_windows ≥ 500 for stable calibration. More is always better.
- [ ] **β condition**: use `SBBTS.suggest_beta(n_time_points=T)`. Never set β manually below this value.
- [ ] **Density smoothness**: avoid features that are nearly discrete (e.g., 0/1 binary, heavily rounded data).
- [ ] **High d**: if d > 20, apply PCA first (`PCAKMeansReducer`). If d > 5, expect training to take longer.
