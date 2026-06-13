# SBBTS — Schrödinger-Bass Bridge for Time Series

A Python library for generating synthetic financial time series that jointly calibrates
drift **and** volatility — fixing the key limitation of Schrödinger Bridge methods,
which fix σ = I and cannot produce stochastic volatility.

Based on: *SBBTS: A Unified Schrödinger–Bass Framework for Synthetic Financial Time Series*,
Alouadi, Loeper, Marsala, Mazhar, Pham — arXiv:2604.07159 (2026).

This is an independent re-implementation of the algorithm described in the paper,
written from scratch as a structured Python library with a scikit-learn-style API.
See [THEORY.md](THEORY.md) for the full mathematical derivation.

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

# X_train: (N_windows, T, d)  — rolling windows of log returns
# e.g. 1000 windows of 252 days of 1 asset
X_train = np.load("spx_windows.npy")   # shape (1000, 252, 1)

beta = SBBTS.suggest_beta(n_time_points=252)   # auto: β·Δt = 5
model = SBBTS(beta=beta, n_steps=5, d_model=128, n_heads=16, n_epochs=1000)
model.fit(X_train)

X_synth = model.sample(n=500)          # (500, 252, 1) — new synthetic paths
X_aug   = model.augment(X_train, factor=200)  # real + 200× synthetic
model.save("sbbts_spx.pt")
```

---

## Data requirements

| Variable | What it is | Minimum | Recommended |
|---|---|---|---|
| `N_windows` | number of rolling windows | ~500 | 1000+ |
| `T` | window length (time steps) | 50 | 252 (1 trading year) |
| `d` | number of assets / factors | 1 | 1–20 (reduce first for d > 20) |

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

## Evaluating output quality

After training, check these metrics in order of importance:

| Metric | What it measures | Target |
|---|---|---|
| **TSTR ratio** | synthetic as substitute for real (AR task) | < 1.05 = excellent |
| **ann_std ratio** | daily vol of synth vs real | 0.8–1.1 |
| **RV mean ratio** | realized variance calibration | 0.8–1.1 |
| **ACF\|r\| sum** | volatility clustering | > 50% of real |
| **Excess kurtosis** | fat tails | within 30% of real |

Quick check:

```python
from sbbts.utils.metrics import compute_stylized_facts

X_synth = model.sample(n=500)
metrics = compute_stylized_facts(X_real, X_synth)
print(metrics)
```

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
    encoder_type="transformer",  # or "signature"
    device=None,                 # auto-detect
    logger=None,                 # SBBTSLogger or W&B/MLflow compatible
)

model.fit(X)                     # X: (N, T, d) ndarray or tensor
model.sample(n)                  # → (n, T, d) ndarray
model.augment(X_real, factor=200)# → (N + 200N, T, d)
model.save("model.pt")
SBBTS.load("model.pt")
SBBTS.suggest_beta(n_time_points, safety_factor=5.0)
```

---

## Saving and resuming

```python
model.save("checkpoint_k3.pt")          # save after any outer iteration
model_resumed = SBBTS.load("checkpoint_k3.pt")
model_resumed.fit(X_train)              # continues training from loaded weights
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
