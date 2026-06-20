# SBBTS — Schrödinger-Bass Bridge for Time Series

**SBBTS** is a Python library for **synthetic financial time series generation** and
**time series data augmentation**. It jointly calibrates drift **and** volatility —
fixing the key limitation of Schrödinger Bridge methods, which fix σ = I and cannot
produce stochastic volatility.

A clean, pip-installable re-implementation of the **Schrödinger-Bass Bridge for Time Series**
framework with a scikit-learn-style API (`.fit()` / `.sample()` / `.augment()`), practical
defaults for financial data, and built-in stylized-fact diagnostics.

Based on: *SBBTS: A Unified Schrödinger–Bass Framework for Synthetic Financial Time Series*,
Alouadi, Loeper, Marsala, Mazhar, Pham — [arXiv:2604.07159](https://arxiv.org/abs/2604.07159) (2026).

See [THEORY.md](THEORY.md) for the full mathematical derivation.

> **Keywords:** SBBTS · Schrödinger-Bass Bridge · Schrodinger Bass Bridge · Schrodinger bridge ·
> Schrödinger bridge time series · synthetic time series · financial time series generation ·
> time series augmentation · synthetic financial data · generative model for time series ·
> optimal transport · diffusion model · quantitative finance · PyTorch

---

## Why this implementation?

The [original paper](https://arxiv.org/abs/2604.07159) ships research scripts.
This repo packages the same algorithm as a proper Python library:

- **scikit-learn-style API** — `model.fit(X)`, `model.sample(n)`, `model.augment(X, factor=200)`
- **One-line install** — `pip install sbbts` (or `pip install -e .` from source)
- **Practical defaults** — `suggest_beta()`, `normalize_input=True`, auto CUDA
- **Built-in diagnostics** — ACF, marginal, QQ, leverage effect, TSTR score, rolling vol
- **Fully documented** — [THEORY.md](THEORY.md) walks through every equation, every parameter
- **No internal dependencies** — pulls SPX data via `yfinance`, runs on CPU or GPU

---

## Install

```bash
git clone https://github.com/JulianoLeperso/sbbts.git
cd sbbts
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0

---

## The 10-line workflow

```python
import numpy as np
from sbbts import SBBTS

# X_train: (N_windows, T, d)  — rolling windows of any stationary time series
# e.g. 1000 windows of 252 steps for 1 feature
X_train = np.load("spx_windows.npy")   # shape (1000, 252, 1)

beta = SBBTS.suggest_beta(n_time_points=252)   # auto: β·Δt = 5
model = SBBTS(beta=beta, n_steps=5, d_model=128, n_heads=16, n_epochs=1000)
model.fit(X_train)

X_synth = model.sample(n=500)          # (500, 252, 1) — new synthetic paths
X_aug   = model.augment(X_train, factor=200)  # real + 200× synthetic
model.save("sbbts_spx.pt")
```

---

## Multi-feature input

SBBTS accepts any multivariate time series — not just financial returns.
Pass `d > 1` features directly as the last axis of `X_train`:

```python
# d=2: log-return + realized variance
X_train = np.stack([windows_log_ret, windows_realized_vol], axis=-1)  # (N, T, 2)

model = SBBTS(beta=SBBTS.suggest_beta(T), n_steps=5)
model.fit(X_train, feature_names=["log_return", "realized_vol"])

X_synth = model.sample(n=500)              # (500, T, 2)
fig = model.diagnose_generic(X_train)      # domain-agnostic diagnostic (no Sharpe/VaR)
```

For a full guide on supported data types, acceptance conditions, and domain-agnostic
workflows → see [GUIDE.md](GUIDE.md).

---

## How it works

Most generative models for time series either fix volatility (standard Schrödinger bridges
set σ = I) or ignore drift (martingale transport). SBBTS fixes both at once by constructing
a diffusion process whose drift *and* volatility are jointly calibrated to the target data.

The key idea is the **Schrödinger-Bass bridge**: a stochastic process that is pinned at
both endpoints of each time interval and whose law matches a target measure. SBBTS decomposes
the full path-space problem into a sequence of these interval-level transport problems,
which makes training tractable — each one is solved by a small score network rather than a
single monolithic model over the whole trajectory.

Training runs `n_steps` outer iterations (K in the paper). Each iteration refines the
approximation: the score network for iteration k is trained against the transport measure
produced by iteration k−1, tightening the fit to the real distribution round by round.
This is why `n_steps` is the primary lever for temporal structure — one iteration captures
marginals well; more iterations progressively sharpen volatility clustering and cross-lag
correlations.

At sampling time, the model draws a Gaussian initial state, maps it into the learned latent
space via the transport map, then runs `n_euler_steps` Euler-Maruyama steps through the
fitted SDE to produce a full synthetic path. The stored normalization (fitted on training
data) is inverted at the end, so outputs are always in the original return scale.

One subtlety worth knowing: financial log returns live at scale ~0.01 while the internal
sampler draws from N(0,1). Without `normalize_input=True` these are 100× apart and training
fails silently — the loss decreases but the generated paths have completely wrong scale.
The flag is on by default and should stay on unless your data is already standardized.

---

## Data requirements

| Variable | What it is | Minimum | Recommended |
|---|---|---|---|
| `N_windows` | number of rolling windows | ~500 | 1000+ |
| `T` | window length (time steps) | 50 | 252 (1 trading year) |
| `d` | number of features / assets | 1 | 1–20 (reduce first for d > 20) |

**Accepted data types (short list):**

| Type | Works? | Notes |
|---|---|---|
| Log-returns | ✅ Validated | Paper benchmark |
| Realized variance / volatility | ✅ OK | Log-transform if log-normal |
| PCA factors | ✅ OK | Already normalized |
| Raw prices | ❌ Avoid | Non-stationary — differentiate first |
| Discrete / categorical | ❌ Not applicable | KL divergence undefined |

Full acceptance conditions and more data types → [GUIDE.md](GUIDE.md).

**How to build windows from daily prices:**

```python
import numpy as np

prices = ...       # 1D array of daily closes
log_ret = np.log(prices[1:] / prices[:-1])   # (N_days-1,)

T = 252
windows = np.lib.stride_tricks.sliding_window_view(log_ret, T)
X_train = windows[:, :, np.newaxis].astype(np.float32)  # (N_win, T, 1)
```

For **d > 20 assets**, reduce first:

```python
from sbbts.utils.dim_reduction import PCAKMeansReducer

reducer = PCAKMeansReducer(n_components=16, n_clusters=3)
X_reduced = reducer.fit_transform(X_high_dim)   # (N, T, 16)
model = SBBTS(beta=..., dim_reducer=reducer)
model.fit(X_high_dim)       # handles reduction internally
X_synth = model.sample(n=500)   # returned in original asset space
```

---

## Choosing β

β controls the drift/volatility tradeoff. The theory requires **β·Δt > 1** on every
interval (Theorem 3.2 in the paper), where Δt = 1/(T−1).

```python
# Safe default: β·Δt = 5 (5× above the existence threshold)
beta = SBBTS.suggest_beta(n_time_points=T, safety_factor=5.0)

# T=252 → Δt ≈ 0.004 → β ≈ 1255
# T=50  → Δt = 0.02  → β ≈ 245
```

If you get `ValueError: Theorem 3.2 condition violated`, your β is too small.
Use `suggest_beta()` to fix it automatically.

---

## Configs

### Lite — fast sanity check (~10–20 min CPU)

```python
LITE_CFG = dict(
    beta=SBBTS.suggest_beta(n_time_points=T),
    n_steps=2,
    d_model=32,
    n_heads=4,
    n_encoder_layers=1,
    n_epochs=300,
    batch_size=128,
    learning_rate=1e-3,
    n_euler_steps=20,
    normalize_input=True,
)
```

Use to verify the pipeline runs and loss decreases before committing to a full run.

### Full — paper quality (~90–120 min GPU, ~7–8 hr CPU)

```python
FULL_CFG = dict(
    beta=SBBTS.suggest_beta(n_time_points=T),
    n_steps=5,
    d_model=128,
    n_heads=16,
    n_encoder_layers=1,
    n_epochs=1000,
    batch_size=128,
    learning_rate=3e-4,
    n_euler_steps=50,
    normalize_input=True,
    grad_clip=0.0,
)
```

---

## Is training working? Read the loss

```
K=1: 4.15 → 0.75   ← large drop = model learning fast (good)
K=2: 0.70 → 0.42   ← transport map refining
K=3: 0.39 → 0.32   ← slowing (expected)
K=4: 0.32 → 0.27   ← near convergence
K=5: 0.24 → 0.22   ← converged (oscillations = noise)
```

**Starting loss** ~4 is normal (= E[||target||²] with zero-init score network).
**Good final loss** for T=252: 0.1–0.3. If it stays above 1.0 after K=5, you need more data.
**Loss flat from epoch 1**: learning rate too high, reduce by 3–10×.
**Loss oscillates wildly**: try `grad_clip=1.0`.

---

## Tuning hyperparameters

Start from the Lite or Full config above. When results are not satisfactory, use this table to decide what to change:

| Symptom | Parameter to change | Direction |
|---|---|---|
| Volatility clustering (ACF of \|r\|) too weak | `n_steps` | Increase (3 → 5 → 8) |
| Wrong volatility level — std ratio off | `normalize_input` | Must be `True`; also check data scale |
| TSTR ratio > 1.1 — model can't substitute real data | `N_windows` (more data) or `n_epochs` | More data first, then more epochs |
| Fat tails not captured | `N_windows` or `d_model` | More data first; wider model second |
| Loss flat from epoch 1 | `learning_rate` | Reduce 3–10× (e.g. 1e-3 → 1e-4) |
| Loss spikes or NaN mid-training | `grad_clip` | Set to `1.0` (or lower) |
| Overfitting — loss drops then rises | `early_stopping_patience` | Set to 50–100 |
| Sampling artifacts / jagged paths | `n_euler_steps` | Increase (50 → 100) |
| Training too slow on CPU | `d_model`, `n_heads`, `n_epochs` | Use Lite config; switch to GPU |

**Which parameters matter most for quality (in order):**

1. `N_windows` — more training windows beats everything else
2. `n_steps` — the single biggest lever for temporal dynamics (volatility clustering, leverage)
3. `beta` — always use `suggest_beta()`; do not hand-tune unless you understand Theorem 3.2
4. `n_epochs` — 1000 is the paper default; 300 is enough to validate the pipeline
5. `d_model` / `n_heads` — network capacity; only matters once data is sufficient
6. `n_euler_steps` — affects sample quality, not training; 50 is fine, 100 if paths look jagged

`t_tilde_offset` (numerical safeguard near the terminal time of each bridge interval) and
`val_fraction` (validation split for early stopping) rarely need to change from their defaults.

---

## Evaluating output quality

After training, check these metrics in order of importance:

| Metric | What it measures | Target |
|---|---|---|
| **TSTR ratio** | synthetic as substitute for real (AR task) | < 1.05 = excellent |
| **ann_std ratio** | daily vol of synth vs real | 0.8–1.1 |
| **RV mean ratio** | realized variance calibration | 0.8–1.1 |
| **ACF\|r\| sum** | volatility clustering | > 50% of real |
| **Excess kurtosis** | fat tails | within 30% of real |

---

## Known limitations

- **Needs ≥ 500 training windows for stable calibration.** With fewer windows the
  model learns the right temporal dynamics (TSTR stays low) but underestimates
  absolute volatility. Use N_DAYS=1260+ (5yr) for full SPX experiments.
- **Volatility clustering** (ACF of |r|) is harder to capture than marginals.
  It improves with more training data and more outer iterations (n_steps=5 vs 2).
- **High-dimensional data (d > 5):** use PCA + k-means first. The transformer
  encoder has O(T²) attention cost; d only affects the final linear layer.
- **CPU training is slow.** T=252, d=1, full config: ~8 hrs on a modern CPU.
  On a single A100: ~15 min (the paper ran on A100 SXM4 40 GB).

---

## API reference

```python
SBBTS(
    beta,                        # required — use suggest_beta()
    n_steps=5,                   # K outer iterations
    d_model=128,                 # transformer / network width
    n_heads=16,                  # attention heads
    n_encoder_layers=1,          # transformer depth
    n_epochs=1000,               # epochs per outer iteration
    batch_size=128,
    learning_rate=3e-4,
    n_euler_steps=50,            # Euler-Maruyama steps (sampling only)
    normalize_input=True,        # strongly recommended for log returns
    grad_clip=0.0,               # 0 = disabled; try 1.0 if loss spikes
    early_stopping_patience=0,   # 0 = disabled
    lr_scheduler="cosine",       # "cosine" or "none"; cosine anneals lr to lr/100
    seed=None,                   # integer for reproducible runs, None = random
    encoder_type="transformer",  # or "signature"
    feature_names=None,          # optional list of d feature labels
    device=None,                 # auto-detect
    logger=None,                 # SBBTSLogger or W&B/MLflow compatible
)

model.fit(
    X,                           # (N, T, d) training data
    feature_names=None,          # overrides __init__
    resume_from_step=0,          # start at outer step k (warm restart)
    checkpoint_dir=None,         # directory for auto-saved checkpoints per outer step
)
model.sample(n)                  # → (n, T, d) ndarray
model.sample_conditional(        # fan chart: n continuations given an observed prefix
    X_prefix,                    # (T_prefix, d) or (B, T_prefix, d)
    n=200,
)                                # → (n, T, d) ndarray
model.sample_batches(n, batch_size=500)  # generator of (B, T, d) batches (avoids OOM)
model.augment(X_real, factor=200)        # → (N + 200N, T, d)
model.evaluate_augmentation(X_real)      # TSTR dict (ratio, trtr_mse, tstr_mse, …)
model.diagnose(X_real)           # financial diagnostic (VaR, Sharpe, leverage…)
model.diagnose_generic(X_real)   # domain-agnostic diagnostic (paths, dist, ACF, corr)
model.save("model.pt")
SBBTS.load("model.pt")
SBBTS.suggest_beta(n_time_points, safety_factor=5.0)
SBBTS.from_config()              # load default.yaml → SBBTS instance
SBBTS.from_config("my.yaml")    # load custom YAML → SBBTS instance
```

**Generic (domain-agnostic) utilities:**

```python
from sbbts import (
    diagnose_generic,          # composite plot — no finance assumptions
    plot_feature_paths,        # one trajectory panel per feature
    plot_feature_marginals,    # one histogram per feature
    plot_feature_acf,          # one ACF panel per feature
    plot_feature_stats,        # synth/real stat-ratio heatmap
    compute_generic_metrics,   # data-agnostic quality metrics
)
```

---

## YAML config

Load the bundled default config (mirrors the paper's full hyperparameters) or supply
your own YAML file:

```python
# Use bundled defaults (sbbts/configs/default.yaml)
model = SBBTS.from_config()

# Use a custom YAML file
model = SBBTS.from_config("my_config.yaml")
model.fit(X_train)
```

Nested YAML sections (`training:`, `network:`, `sampling:`, …) are automatically
flattened, and section-specific aliases are resolved (`K → n_steps`, `N_pi → n_euler_steps`).

---

## Reproducibility

Set `seed` to get identical samples across runs:

```python
model = SBBTS(beta=beta, n_steps=2, seed=42)
model.fit(X_train)

X_synth_a = model.sample(n=100)

# Reload and resample — same result
model2 = SBBTS.load("model.pt")
X_synth_b = model2.sample(n=100)
assert np.allclose(X_synth_a, X_synth_b)  # identical if same seed
```

The seed fixes `torch`, `numpy`, and CUDA RNG states at the start of `fit()`.

---

## Checkpointing

Pass `checkpoint_dir` to `fit()` to save a `.pt` file after every outer step.
Combine with `resume_from_step` to recover from interruptions:

```python
# First run — saves checkpoint_k1.pt, checkpoint_k2.pt, …
model = SBBTS(beta=beta, n_steps=5, d_model=128, n_epochs=1000)
model.fit(X_train, checkpoint_dir="checkpoints/")

# If interrupted at k=2, reload and resume at k=3:
model = SBBTS.load("checkpoints/checkpoint_k2.pt")
model.fit(X_train, checkpoint_dir="checkpoints/", resume_from_step=2)
```

---

## Evaluating output quality

After training, check these metrics in order of importance:

| Metric | What it measures | Target |
|---|---|---|
| **TSTR ratio** | synthetic as substitute for real (AR task) | < 1.05 = excellent |
| **ann_std ratio** | daily vol of synth vs real | 0.8–1.1 |
| **RV mean ratio** | realized variance calibration | 0.8–1.1 |
| **ACF\|r\| sum** | volatility clustering | > 50% of real |
| **Excess kurtosis** | fat tails | within 30% of real |

```python
from sbbts.utils.metrics import compute_metrics, compute_tstr

X_synth = model.sample(n=500)

# Stylized facts dict
metrics = compute_metrics(X_real, X_synth)
print(metrics["ann_std_ratio"])     # target: 0.8–1.1
print(metrics["acf_abs_sum_ratio"]) # target: > 0.5

# TSTR quality score (Train-on-Synthetic, Test-on-Real)
tstr = compute_tstr(X_real, X_synth, ar_order=5)
print(f"TSTR ratio: {tstr['ratio']:.4f}")  # target: < 1.05

# Or via the model (zero-configuration)
result = model.evaluate_augmentation(X_real)
print(f"TSTR ratio: {result['tstr']['ratio']:.4f}")
```

---

## Saving and resuming

```python
model.save("checkpoint_k3.pt")          # save after any outer iteration
model_resumed = SBBTS.load("checkpoint_k3.pt")
model_resumed.fit(X_train, resume_from_step=3)  # continues from outer step 3
```

---

## Citation

If you use this library, please cite the original paper:

```bibtex
@misc{alouadi2026sbbts,
  title   = {SBBTS: A Unified Schr\"odinger--Bass Framework for
             Synthetic Financial Time Series},
  author  = {Alouadi, Alexandre and Loeper, Gr\'egoire and Marsala, C\'elian
             and Mazhar, Othmane and Pham, Huy\^en},
  year    = {2026},
  eprint  = {2604.07159},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2604.07159}
}
```

This library is an independent implementation of the algorithm described in the paper above.
The mathematical framework, decomposition theorem, and neural architecture are the work of
the original authors. The code in this repository was written from scratch as a structured
Python library and is not a redistribution of the authors' original implementation.

The original authors' code is available at: https://github.com/alexouadi/SBBTS
