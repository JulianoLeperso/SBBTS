"""
Visualization utilities for SBBTS.

Diagnostic plots for comparing real vs. synthetic financial time series.
Requires matplotlib: pip install 'sbbts[viz]'

Every function accepts an optional ``logger`` (SBBTSLogger) argument.
When provided, the numeric values behind each plot are written to diagnostics.log
so the full run report is self-contained without reading every figure.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Colour palette ──────────────────────────────────────────────────────────
REAL_C  = "#1565C0"   # deep blue  — real / observed data
SYNTH_C = "#C62828"   # deep red   — synthetic / SBBTS data
GBM_C   = "#2E7D32"   # deep green — GBM or other baseline
# ────────────────────────────────────────────────────────────────────────────


def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("Visualization requires matplotlib: pip install 'sbbts[viz]'")


def plot_acf_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_lag: int = 20,
    ax=None,
    title: str = "Autocorrelation",
    logger=None,
):
    """Side-by-side ACF bars for real vs. synthetic."""
    _require_matplotlib()
    from sbbts.utils.metrics import autocorrelation

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    real_flat  = np.array(real).reshape(-1, np.array(real).shape[-1] if np.array(real).ndim == 3 else 1)
    synth_flat = np.array(synthetic).reshape(-1, np.array(synthetic).shape[-1] if np.array(synthetic).ndim == 3 else 1)

    acf_real  = autocorrelation(real_flat.mean(axis=-1)  if real_flat.ndim  > 1 else real_flat.flatten(),  max_lag)
    acf_synth = autocorrelation(synth_flat.mean(axis=-1) if synth_flat.ndim > 1 else synth_flat.flatten(), max_lag)
    lags = np.arange(max_lag + 1)

    ax.bar(lags - 0.2, acf_real,  width=0.35, label="Real",      color=REAL_C,  alpha=0.85)
    ax.bar(lags + 0.2, acf_synth, width=0.35, label="Synthetic", color=SYNTH_C, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if logger is not None:
        logger.section(f"ACF Comparison — {title}")
        spot_lags = [l for l in [1, 5, 10, 20] if l <= max_lag]
        rows = [
            [lag,
             f"{float(acf_real[lag]):.4f}",
             f"{float(acf_synth[lag]):.4f}",
             f"{float(acf_synth[lag] - acf_real[lag]):+.4f}"]
            for lag in spot_lags
        ]
        logger.write_table(["Lag", "Real ACF", "Synth ACF", "Diff"], rows)
        sum_r = float(np.sum(np.abs(acf_real[1:])))
        sum_s = float(np.sum(np.abs(acf_synth[1:])))
        logger.write(f"  Sum |ACF| lags 1-{max_lag}: real={sum_r:.4f}  synth={sum_s:.4f}"
                     f"  ratio={sum_s/(sum_r+1e-10):.3f}")

    return ax


def plot_marginal_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
    ax=None,
    title: str = "Return Distribution",
    n_bins: int = 60,
    logger=None,
):
    """Overlaid histograms of marginal return distributions."""
    _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    r = np.array(real).flatten()
    s = np.array(synthetic).flatten()

    x_lo = np.percentile(r, 0.5)
    x_hi = np.percentile(r, 99.5)
    bins = np.linspace(x_lo, x_hi, n_bins)
    ax.hist(r, bins=bins, density=True, alpha=0.55, label="Real",      color=REAL_C)
    ax.hist(s, bins=bins, density=True, alpha=0.55, label="Synthetic", color=SYNTH_C)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if logger is not None:
        try:
            from scipy.stats import kurtosis as _kurt, skew as _skew
            skew_r, skew_s   = float(_skew(r)),          float(_skew(s))
            kurt_r, kurt_s   = float(_kurt(r,fisher=True)), float(_kurt(s,fisher=True))
        except ImportError:
            skew_r = skew_s = kurt_r = kurt_s = float("nan")

        logger.section(f"Marginal Distribution — {title}")
        rows = [
            ["mean",     f"{float(r.mean()):.6f}",          f"{float(s.mean()):.6f}"],
            ["std",      f"{float(r.std(ddof=1)):.6f}",     f"{float(s.std(ddof=1)):.6f}"],
            ["skew",     f"{skew_r:.4f}",                    f"{skew_s:.4f}"],
            ["kurtosis", f"{kurt_r:.4f}",                    f"{kurt_s:.4f}"],
            ["min",      f"{float(r.min()):.6f}",            f"{float(s.min()):.6f}"],
            ["p1",       f"{float(np.percentile(r,1)):.6f}", f"{float(np.percentile(s,1)):.6f}"],
            ["p5",       f"{float(np.percentile(r,5)):.6f}", f"{float(np.percentile(s,5)):.6f}"],
            ["p95",      f"{float(np.percentile(r,95)):.6f}",f"{float(np.percentile(s,95)):.6f}"],
            ["p99",      f"{float(np.percentile(r,99)):.6f}",f"{float(np.percentile(s,99)):.6f}"],
            ["max",      f"{float(r.max()):.6f}",            f"{float(s.max()):.6f}"],
        ]
        logger.write_table(["Stat", "Real", "Synth"], rows)
        std_ratio = float(s.std(ddof=1)) / (float(r.std(ddof=1)) + 1e-10)
        logger.write(f"  std ratio (synth/real) : {std_ratio:.3f}  (target ≈ 1.0)")

    return ax


def plot_correlation_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_assets: int = 30,
    axes=None,
    logger=None,
):
    """Heatmaps of correlation matrices side by side."""
    _require_matplotlib()

    real      = np.array(real)
    synthetic = np.array(synthetic)

    if real.ndim == 3:
        real      = real.reshape(-1, real.shape[-1])
        synthetic = synthetic.reshape(-1, synthetic.shape[-1])

    d      = min(real.shape[-1], max_assets)
    corr_r = np.corrcoef(real[:, :d].T)
    corr_s = np.corrcoef(synthetic[:, :d].T)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, corr, label in zip(axes, [corr_r, corr_s], ["Real", "Synthetic"]):
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"{label} Correlation Matrix")
        plt.colorbar(im, ax=ax)

    if logger is not None:
        logger.section("Asset Correlation Matrix")
        logger.write(f"  d (assets shown) : {d}")
        if d > 1:
            idx   = np.triu_indices(d, k=1)
            off_r = corr_r[idx]
            off_s = corr_s[idx]
            diff  = off_s - off_r
            logger.write(f"  Real  off-diag — mean: {float(off_r.mean()):.4f}  std: {float(off_r.std()):.4f}"
                         f"  min: {float(off_r.min()):.4f}  max: {float(off_r.max()):.4f}")
            logger.write(f"  Synth off-diag — mean: {float(off_s.mean()):.4f}  std: {float(off_s.std()):.4f}"
                         f"  min: {float(off_s.min()):.4f}  max: {float(off_s.max()):.4f}")
            logger.write(f"  Mean abs diff          : {float(np.abs(diff).mean()):.4f}")
            logger.write(f"  Frobenius norm of diff : {float(np.linalg.norm(corr_s - corr_r)):.4f}")
        else:
            logger.write("  d=1 (univariate) — trivial 1×1 matrix, no off-diagonal elements")

    return axes


def plot_sample_paths(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_paths: int = 5,
    dim: int = 0,
    axes=None,
    logger=None,
):
    """Overlay sample trajectories from real and synthetic data."""
    _require_matplotlib()

    real      = np.array(real)
    synthetic = np.array(synthetic)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

    for data, ax, label, color in zip(
        [real, synthetic], axes, ["Real", "Synthetic"], [REAL_C, SYNTH_C]
    ):
        for i in range(min(n_paths, len(data))):
            path = data[i, :, dim] if data.ndim == 3 else data[i]
            ax.plot(path, alpha=0.65, linewidth=0.9, color=color)
        ax.set_title(f"{label} Paths (dim={dim})")
        ax.set_xlabel("Time step")
        ax.grid(True, alpha=0.3)

    if logger is not None:
        r_vals = (real[:, :, dim] if real.ndim == 3 else real).flatten()
        s_vals = (synthetic[:, :, dim] if synthetic.ndim == 3 else synthetic).flatten()
        logger.section("Sample Paths Statistics")
        logger.write(f"  N real  windows : {len(real)}   N synth windows : {len(synthetic)}")
        rows = [
            ["mean", f"{float(r_vals.mean()):.6f}",         f"{float(s_vals.mean()):.6f}"],
            ["std",  f"{float(r_vals.std(ddof=1)):.6f}",    f"{float(s_vals.std(ddof=1)):.6f}"],
            ["min",  f"{float(r_vals.min()):.6f}",           f"{float(s_vals.min()):.6f}"],
            ["max",  f"{float(r_vals.max()):.6f}",           f"{float(s_vals.max()):.6f}"],
        ]
        logger.write_table(["Stat", "Real", "Synth"], rows)
        std_ratio = float(s_vals.std(ddof=1)) / (float(r_vals.std(ddof=1)) + 1e-10)
        logger.write(f"  std ratio (synth/real) : {std_ratio:.3f}")

    return axes


def plot_risk_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    ax=None,
    logger=None,
):
    """Bar chart of VaR / ES / Sharpe for real vs. synthetic."""
    _require_matplotlib()
    from sbbts.utils.metrics import compute_all_risk_metrics

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    keys   = ["var_95", "var_99", "es_95", "es_99", "sharpe"]
    labels = ["VaR 95%", "VaR 99%", "ES 95%", "ES 99%", "Sharpe"]
    rm_r = compute_all_risk_metrics(np.array(real).flatten())
    rm_s = compute_all_risk_metrics(np.array(synthetic).flatten())

    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, [rm_r[k] for k in keys], w, label="Real",      color=REAL_C,  alpha=0.85)
    ax.bar(x + w / 2, [rm_s[k] for k in keys], w, label="Synthetic", color=SYNTH_C, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Risk Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if logger is not None:
        logger.section("Risk Metrics")
        all_keys    = ["ann_return", "ann_std", "sharpe", "var_95", "var_99", "es_95", "es_99"]
        all_labels  = ["Ann Return", "Ann Std",  "Sharpe", "VaR 95%", "VaR 99%", "ES 95%", "ES 99%"]
        rows = []
        for k, label in zip(all_keys, all_labels):
            rv  = rm_r.get(k, float("nan"))
            sv  = rm_s.get(k, float("nan"))
            ratio = sv / rv if rv != 0 and k != "ann_return" else "—"
            ratio_s = f"{ratio:.3f}" if isinstance(ratio, float) else ratio
            rows.append([label, f"{rv:.6f}", f"{sv:.6f}", ratio_s])
        logger.write_table(["Metric", "Real", "Synth", "Ratio"], rows)

    return ax


def plot_lag_corr_matrix(
    real: np.ndarray,
    synthetic: np.ndarray,
    axes=None,
    logger=None,
):
    """
    T×T cross-time correlation heatmaps: real | synthetic | difference.

    Entry (i, j) = Corr(window[:, i], window[:, j]) across all N windows.

    Args:
        real: shape (N, T, d) or (N, T) — uses dimension 0
        synthetic: same shape
        axes: optional list of 3 matplotlib Axes

    Returns:
        list of 3 Axes
    """
    _require_matplotlib()

    def _corr(arr):
        w = np.array(arr)
        if w.ndim == 3:
            w = w[:, :, 0]
        return np.corrcoef(w.T)

    corr_r = _corr(real)
    corr_s = _corr(synthetic)
    diff   = corr_s - corr_r

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(16, 4))

    kw = dict(vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    for ax, mat, label in zip(axes[:2], [corr_r, corr_s], ["Real", "Synthetic"]):
        im = ax.imshow(mat, **kw)
        ax.set_title(f"{label} — cross-time corr")
        ax.set_xlabel("day in window")
        ax.set_ylabel("day in window")
        plt.colorbar(im, ax=ax)

    im = axes[2].imshow(diff, vmin=-0.3, vmax=0.3, cmap="PiYG", aspect="auto")
    axes[2].set_title("Difference (synth − real)\n≈0 = perfect")
    axes[2].set_xlabel("day in window")
    axes[2].set_ylabel("day in window")
    plt.colorbar(im, ax=axes[2])

    if logger is not None:
        diff_abs = np.abs(diff)
        T = corr_r.shape[0]
        logger.section("Cross-time Correlation Matrix")
        logger.write(f"  Matrix size  : {T}×{T}")
        logger.write(f"  Mean abs error  (synth − real) : {float(diff_abs.mean()):.4f}")
        logger.write(f"  Max  abs error                 : {float(diff_abs.max()):.4f}")
        logger.write(f"  Frobenius norm of diff         : {float(np.linalg.norm(diff)):.4f}")
        # Off-diagonal mean absolute correlation
        off_idx = ~np.eye(T, dtype=bool)
        logger.write(f"  Off-diag mean |corr| real      : {float(np.abs(corr_r[off_idx]).mean()):.4f}")
        logger.write(f"  Off-diag mean |corr| synth     : {float(np.abs(corr_s[off_idx]).mean()):.4f}")
        # First-order autocorrelation band (super-diagonal)
        if T > 1:
            lag1_r = float(np.mean([corr_r[i, i+1] for i in range(T-1)]))
            lag1_s = float(np.mean([corr_s[i, i+1] for i in range(T-1)]))
            logger.write(f"  Lag-1 band mean corr  real  : {lag1_r:.4f}")
            logger.write(f"  Lag-1 band mean corr  synth : {lag1_s:.4f}")

    return axes


def plot_qq(
    real: np.ndarray,
    synthetic: np.ndarray,
    ax=None,
    n_quantiles: int = 200,
    logger=None,
):
    """
    QQ-plot of real and synthetic returns vs Normal distribution.

    Fat tails appear as S-curve deviations from the diagonal.

    Args:
        real: 1-D or flattenable return array
        synthetic: 1-D or flattenable return array
        ax: optional matplotlib Axes
        n_quantiles: number of quantile points

    Returns:
        Axes
    """
    _require_matplotlib()
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("plot_qq requires scipy: pip install scipy")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    def _qq(series):
        s = np.array(series).flatten()
        s = (s - s.mean()) / s.std()
        probs  = np.linspace(0.01, 0.99, n_quantiles)
        q_data = np.quantile(s, probs)
        q_norm = stats.norm.ppf(probs)
        return q_norm, q_data

    q_n, q_r = _qq(real)
    q_n, q_s = _qq(synthetic)

    ax.plot(q_n, q_n, "k--", lw=1,   label="Normal")
    ax.plot(q_n, q_r, "o-",  color=REAL_C,  ms=3, lw=1.2, label="Real")
    ax.plot(q_n, q_s, "s--", color=SYNTH_C, ms=3, lw=1.2, label="Synthetic")
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Data quantiles (standardised)")
    ax.set_title("QQ-plot vs Normal\nS-curve = fat tails")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if logger is not None:
        probs_spot = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        q_norm_spot = stats.norm.ppf(probs_spot)

        def _q_at(series, probs):
            s = np.array(series).flatten()
            s = (s - s.mean()) / s.std()
            return np.quantile(s, probs)

        qr_spot = _q_at(real,      probs_spot)
        qs_spot = _q_at(synthetic, probs_spot)

        logger.section("QQ-plot vs Normal")
        rows = [
            [f"{p:.0%}",
             f"{float(qn):.4f}",
             f"{float(qr):.4f}",
             f"{float(qs):.4f}",
             f"{float(qr-qn):+.4f}",
             f"{float(qs-qn):+.4f}"]
            for p, qn, qr, qs in zip(probs_spot, q_norm_spot, qr_spot, qs_spot)
        ]
        logger.write_table(
            ["Pctile", "Normal q", "Real q", "Synth q", "Real dev", "Synth dev"],
            rows,
        )
        max_dev_r = float(np.max(np.abs(q_r - q_n)))
        max_dev_s = float(np.max(np.abs(q_s - q_n)))
        logger.write(f"  Max |deviation from Normal|: real={max_dev_r:.4f}  synth={max_dev_s:.4f}")
        logger.write(f"  (Large deviation in tails = fat-tailed distribution — expected for SPX)")

    return ax


def plot_acf_vol(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_lag: int = 20,
    axes=None,
    logger=None,
):
    """
    ACF of |returns| and ACF of returns² side by side.

    Both measure volatility clustering; r² is more sensitive to crashes.

    Args:
        real: return series (1-D or flattenable)
        synthetic: return series (1-D or flattenable)
        max_lag: maximum lag
        axes: optional list/array of 2 Axes

    Returns:
        list of 2 Axes
    """
    _require_matplotlib()
    from sbbts.utils.metrics import autocorrelation

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4))

    r = np.array(real).flatten()
    s = np.array(synthetic).flatten()
    lags = np.arange(max_lag + 1)

    acf_results = {}
    for ax, fn, title, key in zip(
        axes,
        [np.abs, lambda x: x ** 2],
        ["ACF of |returns|", "ACF of returns²"],
        ["abs", "sq"],
    ):
        acf_r = autocorrelation(fn(r), max_lag)
        acf_s = autocorrelation(fn(s), max_lag)
        acf_results[key] = (acf_r, acf_s)
        ax.plot(lags, acf_r, "o-",  color=REAL_C,  ms=3, lw=1.2, label="Real")
        ax.plot(lags, acf_s, "s--", color=SYNTH_C, ms=3, lw=1.2, label="Synthetic")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("lag (days)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if logger is not None:
        logger.section("Volatility Clustering — ACF of |returns| and returns²")
        for key, label in [("abs", "ACF |r|"), ("sq", "ACF r²")]:
            acf_r, acf_s = acf_results[key]
            spot = [l for l in [1, 5, 10, 15, 20] if l <= max_lag]
            rows = [
                [lag,
                 f"{float(acf_r[lag]):.4f}",
                 f"{float(acf_s[lag]):.4f}",
                 f"{float(acf_s[lag] - acf_r[lag]):+.4f}"]
                for lag in spot
            ]
            logger.write(f"\n  {label}:")
            logger.write_table(["Lag", "Real", "Synth", "Diff"], rows)
            score_r = float(np.sum(acf_r[1:6]))
            score_s = float(np.sum(acf_s[1:6]))
            ratio   = score_s / (score_r + 1e-10)
            flag    = "  ⚠ LOW" if ratio < 0.4 else ""
            logger.write(f"  Sum lags 1-5: real={score_r:.4f}  synth={score_s:.4f}"
                         f"  ratio={ratio:.3f}{flag}")

    return axes


def plot_rolling_vol(
    real: np.ndarray,
    synthetic: np.ndarray,
    roll: int = 21,
    n_show: int = 300,
    annualize: bool = True,
    axes=None,
    logger=None,
):
    """
    Rolling volatility time series and distribution of vol levels.

    Pass the **raw 1-D return series** (not overlapping windows) for correct results.
    If 3-D windows are passed they are flattened, which may repeat values.

    Args:
        real: 1-D return series or flattenable array
        synthetic: same shape
        roll: rolling window size in days (default 21 ≈ 1 trading month)
        n_show: number of observations to show in time-series panel
        annualize: if True, multiply rolling std by √252 for annualised vol
        axes: optional list/array of 2 Axes

    Returns:
        list of 2 Axes
    """
    _require_matplotlib()

    factor = np.sqrt(252) if annualize else 1.0
    ylabel = "annualised vol (√252 factor)" if annualize else "rolling std"

    def _rv(arr):
        import pandas as pd
        flat = np.array(arr).flatten()
        return pd.Series(flat).rolling(roll).std().dropna().values * factor

    rv_r = _rv(real)
    rv_s = _rv(synthetic)

    if logger is not None:
        pct   = 100.0
        label = "annualised" if annualize else "daily"
        ratio = float(rv_s.mean()) / (float(rv_r.mean()) + 1e-10)
        flag  = "  ⚠ INFLATED" if ratio > 3 else ("  ⚠ LOW" if ratio < 0.4 else "")
        logger.section(f"Rolling Volatility ({roll}-day, {label})")
        logger.write(f"  Real  — mean: {rv_r.mean()*pct:.2f}%  std: {rv_r.std()*pct:.2f}%"
                     f"  min: {rv_r.min()*pct:.2f}%  max: {rv_r.max()*pct:.2f}%"
                     f"  p25: {np.percentile(rv_r,25)*pct:.2f}%  p75: {np.percentile(rv_r,75)*pct:.2f}%")
        logger.write(f"  Synth — mean: {rv_s.mean()*pct:.2f}%  std: {rv_s.std()*pct:.2f}%"
                     f"  min: {rv_s.min()*pct:.2f}%  max: {rv_s.max()*pct:.2f}%"
                     f"  p25: {np.percentile(rv_s,25)*pct:.2f}%  p75: {np.percentile(rv_s,75)*pct:.2f}%")
        logger.write(f"  Vol ratio (synth/real): {ratio:.3f}x{flag}")
        logger.write(f"  IQR real  : [{np.percentile(rv_r,25)*pct:.2f}%, {np.percentile(rv_r,75)*pct:.2f}%]")
        logger.write(f"  IQR synth : [{np.percentile(rv_s,25)*pct:.2f}%, {np.percentile(rv_s,75)*pct:.2f}%]")

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(rv_r[:n_show], color=REAL_C,  lw=0.9, label="Real",      alpha=0.9)
    axes[0].plot(rv_s[:n_show], color=SYNTH_C, lw=0.9, label="Synthetic", alpha=0.9)
    axes[0].set_title(f"{roll}-day rolling volatility (first {n_show} obs)")
    axes[0].set_xlabel("observation")
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    v_hi = np.percentile(np.concatenate([rv_r, rv_s]), 99)
    bins = np.linspace(0, v_hi, 50)
    axes[1].hist(rv_r, bins=bins, density=True, alpha=0.55, label="Real",      color=REAL_C)
    axes[1].hist(rv_s, bins=bins, density=True, alpha=0.55, label="Synthetic", color=SYNTH_C)
    axes[1].set_xlim(0, v_hi)
    axes[1].set_title("Distribution of rolling-vol levels")
    axes[1].set_xlabel(ylabel)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    return axes


def plot_cluster_diagnostics(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_clusters: int = 3,
    axes=None,
    logger=None,
):
    """
    Per-regime return distribution and ACF of |returns|.

    K-means is fitted on 5 summary features of each real window.
    Synthetic windows are assigned to the same clusters for comparison.

    Args:
        real: shape (N, T, 1) or (N, T)
        synthetic: shape (M, T, 1) or (M, T)
        n_clusters: number of K-means clusters (default 3)
        axes: optional (n_clusters, 2) array of Axes

    Returns:
        (n_clusters, 2) array of Axes
    """
    _require_matplotlib()
    from sbbts.utils.metrics import autocorrelation

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("plot_cluster_diagnostics requires scikit-learn: pip install scikit-learn")

    r = np.array(real)
    s = np.array(synthetic)
    if r.ndim == 3:
        r = r[:, :, 0]
    if s.ndim == 3:
        s = s[:, :, 0]

    def _features(w):
        return np.stack([
            w.mean(axis=1),
            w.std(axis=1),
            w.min(axis=1),
            w.max(axis=1),
            (w < 0).mean(axis=1),
        ], axis=1)

    scaler = StandardScaler()
    feat_r = scaler.fit_transform(_features(r))
    feat_s = scaler.transform(_features(s))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_r = km.fit_predict(feat_r)
    labels_s = km.predict(feat_s)

    cluster_vol = [r[labels_r == k].std() for k in range(n_clusters)]
    order       = np.argsort(cluster_vol)
    names       = {order[i]: n for i, n in enumerate(["Low-vol", "Mid-vol", "High-vol"][:n_clusters])}

    if logger is not None:
        logger.section(f"Cluster Diagnostics (K-means, k={n_clusters})")
        logger.write(f"  n_real_total={len(r)}  n_synth_total={len(s)}")
        rows = []
        for k in range(n_clusters):
            name   = names[k]
            r_k    = r[labels_r == k]
            s_k    = s[labels_s == k]
            n_r    = len(r_k)
            n_s    = len(s_k)
            rv     = float(r_k.std()) if n_r > 0 else float("nan")
            sv     = float(s_k.std()) if n_s > 1 else float("nan")
            pct_r  = f"{100*n_r/len(r):.1f}%"
            pct_s  = f"{100*n_s/len(s):.1f}%" if len(s) > 0 else "N/A"
            status = "OK" if n_s >= 5 else "  ⚠ FEW SYNTH"
            rows.append([name, n_r, pct_r, n_s, pct_s, f"{rv:.5f}", f"{sv:.5f}" if n_s > 1 else "N/A", status])
        logger.write_table(
            ["Cluster", "n_real", "%_real", "n_synth", "%_synth", "real_vol", "synth_vol", "status"],
            rows,
        )
        logger.write("")
        logger.write("  Diagnosis: if n_synth=0 for low/mid-vol clusters, the model is generating")
        logger.write("  out-of-distribution high-vol samples — likely undertrained (too few epochs")
        logger.write("  or capacity too small). Compare with a full-config run to confirm.")

    if axes is None:
        fig, axes = plt.subplots(n_clusters, 2, figsize=(13, 4 * n_clusters))
        fig.suptitle("Per-regime diagnostics: real vs synthetic", fontsize=13, y=1.01)

    for k in range(n_clusters):
        name   = names[k]
        r_rets = r[labels_r == k].flatten()
        s_rets = s[labels_s == k].flatten()

        n_s = len(s_rets) // max(r.shape[1], 1)
        warn = "  ⚠ few synth windows" if n_s < 5 else ""

        ax_hist, ax_acf = axes[k, 0], axes[k, 1]

        x_lo = np.percentile(r_rets, 0.5)
        x_hi = np.percentile(r_rets, 99.5)
        bins = np.linspace(x_lo, x_hi, 50)

        ax_hist.hist(r_rets, bins=bins, density=True, alpha=0.6, label="Real",      color=REAL_C)
        if len(s_rets) > 2:
            ax_hist.hist(s_rets, bins=bins, density=True, alpha=0.6, label="Synthetic", color=SYNTH_C)
        ax_hist.set_xlim(x_lo, x_hi)
        ax_hist.set_title(
            f"{name} — distribution"
            f"  (n_real={len(r_rets)//r.shape[1]}, n_synth={n_s}{warn})"
        )
        ax_hist.set_xlabel("log return")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        acf_r = autocorrelation(np.abs(r_rets), max_lag=15)
        acf_s = autocorrelation(np.abs(s_rets), max_lag=15) if len(s_rets) > 15 else np.zeros(16)
        lags  = np.arange(16)
        ax_acf.bar(lags - 0.2, acf_r, 0.38, label="Real",      color=REAL_C,  alpha=0.85)
        ax_acf.bar(lags + 0.2, acf_s, 0.38, label="Synthetic", color=SYNTH_C, alpha=0.85)
        ax_acf.axhline(0, color="k", lw=0.5)
        ax_acf.set_title(f"{name} — ACF of |returns|")
        ax_acf.set_xlabel("lag (days)")
        ax_acf.legend()
        ax_acf.grid(True, alpha=0.3)

    return axes


def plot_leverage_effect(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_lag: int = 10,
    axes=None,
    logger=None,
):
    """
    Leverage effect: cross-correlation Corr(r_t, r²_{t+k}) for k in [-max_lag, +max_lag].

    The leverage effect: negative returns today predict higher future volatility.
    This appears as strongly negative Corr(r_t, r²_{t+k}) for k > 0.
    GBM produces ~0 at all lags. SBBTS should track real.

    Args:
        real: 1-D or flattenable return array
        synthetic: 1-D or flattenable return array
        max_lag: maximum lag in each direction
        axes: optional list/array of 2 Axes

    Returns:
        list of 2 Axes
    """
    _require_matplotlib()

    def _leverage_cc(rets):
        r = np.array(rets).flatten()
        r2 = r ** 2
        r_z  = (r  - r.mean())  / (r.std()  + 1e-12)
        r2_z = (r2 - r2.mean()) / (r2.std() + 1e-12)
        lags  = np.arange(-max_lag, max_lag + 1)
        corrs = []
        for lag in lags:
            if lag == 0:
                c = float(np.mean(r_z * r2_z))
            elif lag > 0:
                c = float(np.mean(r_z[:-lag] * r2_z[lag:]))
            else:
                pl = -lag
                c = float(np.mean(r_z[pl:] * r2_z[:-pl]))
            corrs.append(c)
        return lags, np.array(corrs)

    lags, cc_r = _leverage_cc(real)
    _,    cc_s = _leverage_cc(synthetic)

    if logger is not None:
        logger.section("Leverage Effect — Corr(r_t, r²_{t+k})")
        spot = [k for k in range(max_lag + 1) if k <= max_lag]
        rows = []
        for k in spot:
            idx   = np.where(lags == k)[0][0]
            rv    = float(cc_r[idx])
            sv    = float(cc_s[idx])
            flag  = "✓" if k == 0 or (rv < 0) == (sv < 0) else "✗ sign mismatch"
            rows.append([k, f"{rv:.4f}", f"{sv:.4f}", f"{sv-rv:+.4f}", flag])
        logger.write_table(["Lag k", "Real", "Synth", "Diff", "Sign"], rows)
        # k=1 sign verdict
        idx1  = np.where(lags == 1)[0][0]
        sign_ok = (cc_r[idx1] < 0) == (cc_s[idx1] < 0)
        logger.write(f"\n  Leverage sign at k=1  : real={float(cc_r[idx1]):.4f}  "
                     f"synth={float(cc_s[idx1]):.4f}  "
                     f"sign_match={'YES' if sign_ok else 'NO ⚠'}")
        logger.write("  (Real equity: should be negative at k>0 — losses predict higher future vol)")

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4))

    w = 0.38
    axes[0].bar(lags - w / 2, cc_r, w, label="Real",      color=REAL_C,  alpha=0.85)
    axes[0].bar(lags + w / 2, cc_s, w, label="Synthetic", color=SYNTH_C, alpha=0.85)
    axes[0].axhline(0, color="k",  lw=0.6)
    axes[0].axvline(0, color="k",  lw=0.6, ls="--", alpha=0.5)
    axes[0].set_xlabel("lag k")
    axes[0].set_ylabel("Corr(r_t , r²_{t+k})")
    axes[0].set_title("Leverage Effect  (all lags)\nNegative k>0 = future vol rises after losses")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    pos = lags >= 0
    axes[1].plot(lags[pos], cc_r[pos], "o-",  color=REAL_C,  ms=5, lw=1.5, label="Real")
    axes[1].plot(lags[pos], cc_s[pos], "s--", color=SYNTH_C, ms=5, lw=1.5, label="Synthetic")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].fill_between(lags[pos], cc_r[pos], 0, where=cc_r[pos] < 0,
                         color=REAL_C, alpha=0.12, label="_nolegend_")
    axes[1].set_xlabel("lag k  (k ≥ 0)")
    axes[1].set_ylabel("Corr(r_t , r²_{t+k})")
    axes[1].set_title("Leverage Effect — forward lags\nReal equity: should be negative here")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    return axes


def plot_signature_moments(
    real: np.ndarray,
    synthetic: np.ndarray,
    depth: int = 2,
    axes=None,
    logger=None,
):
    """
    Truncated path signature moments: compare real vs synthetic.

    For d=1, depth=2 gives 3 interpretable components per window:
      S¹  = X_T - X_0               (total return of the window)
      S¹¹ = Σ_{t<T} S¹_t · ΔX_t+1 (iterated integral — momentum profile)
      RV  = (S¹)² − 2·S¹¹           (realized variance; Chen's identity)

    Args:
        real: shape (N, T, d) or (N, T)
        synthetic: same shape
        depth: signature truncation depth (1 or 2)
        axes: optional array of Axes

    Returns:
        axes array
    """
    _require_matplotlib()
    try:
        import torch
        from sbbts.nn.signature_encoder import _incremental_signatures
    except ImportError:
        raise ImportError("plot_signature_moments requires torch and the sbbts package")

    r = np.array(real)
    s = np.array(synthetic)
    if r.ndim == 2:
        r = r[:, :, np.newaxis]
    if s.ndim == 2:
        s = s[:, :, np.newaxis]

    d = r.shape[-1]

    r_t = torch.tensor(r, dtype=torch.float32)
    s_t = torch.tensor(s, dtype=torch.float32)

    sig_r = _incremental_signatures(r_t, depth)[:, -1, :].numpy()  # (N, sig_dim)
    sig_s = _incremental_signatures(s_t, depth)[:, -1, :].numpy()  # (M, sig_dim)

    if depth == 1:
        names = [f"S¹[{i}]  (total return, dim {i})" for i in range(d)]
        vals_r = [sig_r[:, i] for i in range(d)]
        vals_s = [sig_s[:, i] for i in range(d)]
    else:
        names_l1 = [f"S¹[{i}]  total return, dim {i}" for i in range(d)]
        names_l2 = [f"S¹¹[{i},{j}]  iterated integral" for i in range(d) for j in range(d)]
        all_names = names_l1 + names_l2
        all_r = [sig_r[:, i] for i in range(len(all_names))]
        all_s = [sig_s[:, i] for i in range(len(all_names))]

        rv_names = [f"RV[{i}]  ≈ (S¹)² − 2·S¹¹  (realized variance)" for i in range(d)]
        rv_r = [(sig_r[:, i] ** 2) - 2 * sig_r[:, d + i * d + i] for i in range(d)]
        rv_s = [(sig_s[:, i] ** 2) - 2 * sig_s[:, d + i * d + i] for i in range(d)]

        names  = all_names + rv_names
        vals_r = all_r     + rv_r
        vals_s = all_s     + rv_s

    if logger is not None:
        logger.section("Path Signature Moments")
        short = [n.split("(")[0].strip()[:35] for n in names]
        rows = []
        for ci, label in enumerate(short):
            rv    = vals_r[ci]
            sv    = vals_s[ci]
            ratio = abs(float(sv.mean())) / (abs(float(rv.mean())) + 1e-10)
            flag  = "  ⚠ LARGE" if ratio > 5 else ""
            rows.append([
                label,
                f"{float(rv.mean()):.6f}", f"{float(rv.std()):.6f}",
                f"{float(sv.mean()):.6f}", f"{float(sv.std()):.6f}",
                f"{ratio:.2f}x{flag}",
            ])
        logger.write_table(
            ["Moment", "Real mean", "Real std", "Synth mean", "Synth std", "ratio"],
            rows,
        )

    n_comp = len(names)
    n_cols = min(n_comp, 3)
    n_rows = (n_comp + n_cols - 1) // n_cols

    if axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle("Path Signature Moments — real vs synthetic", fontsize=12)

    axes_flat = np.array(axes).flatten()

    for ci in range(n_comp):
        ax = axes_flat[ci]
        rv = vals_r[ci]
        sv = vals_s[ci]

        x_lo = np.percentile(np.concatenate([rv, sv]), 1)
        x_hi = np.percentile(np.concatenate([rv, sv]), 99)
        bins = np.linspace(x_lo, x_hi, 45)
        ax.hist(rv, bins=bins, density=True, alpha=0.55, label="Real",      color=REAL_C)
        ax.hist(sv, bins=bins, density=True, alpha=0.55, label="Synthetic", color=SYNTH_C)
        ax.set_title(names[ci], fontsize=9)
        ax.set_xlim(x_lo, x_hi)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ci in range(n_comp, len(axes_flat)):
        axes_flat[ci].set_visible(False)

    return axes


def tstr_score(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    ar_lags: int = 5,
    logger=None,
) -> dict:
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.

    Fits an AR(ar_lags) linear model on synthetic data and evaluates MSE on real
    held-out windows. Compares to TRTR (train on real, test on real) baseline.

    A ratio close to 1.0 means synthetic data is an adequate training substitute.
    Ratio >> 1 means synthetic data misleads the learner.

    Args:
        X_real: shape (N, T, 1) or (N, T)
        X_synth: shape (M, T, 1) or (M, T)
        ar_lags: number of AR lags (default 5)
        logger: optional SBBTSLogger

    Returns:
        dict with keys: trtr_mse, tstr_mse, ratio, n_real_train, n_real_test, n_synth
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        raise ImportError("tstr_score requires scikit-learn: pip install scikit-learn")

    r = np.array(X_real)
    s = np.array(X_synth)
    if r.ndim == 3:
        r = r[:, :, 0]
    if s.ndim == 3:
        s = s[:, :, 0]

    def _ar_dataset(windows, lags):
        X_feat, y_feat = [], []
        for row in windows:
            for t in range(lags, len(row)):
                X_feat.append(row[t - lags:t])
                y_feat.append(row[t])
        return np.array(X_feat), np.array(y_feat)

    split = int(0.8 * len(r))
    X_r_tr, y_r_tr = _ar_dataset(r[:split], ar_lags)
    X_r_te, y_r_te = _ar_dataset(r[split:],  ar_lags)
    X_s,    y_s    = _ar_dataset(s,           ar_lags)

    lr_r = LinearRegression().fit(X_r_tr, y_r_tr)
    lr_s = LinearRegression().fit(X_s,    y_s)

    trtr = float(np.mean((lr_r.predict(X_r_te) - y_r_te) ** 2))
    tstr = float(np.mean((lr_s.predict(X_r_te) - y_r_te) ** 2))
    ratio = tstr / trtr if trtr > 0 else float("inf")

    result = {
        "trtr_mse":     trtr,
        "tstr_mse":     tstr,
        "ratio":        ratio,
        "n_real_train": len(y_r_tr),
        "n_real_test":  len(y_r_te),
        "n_synth":      len(y_s),
    }

    if logger is not None:
        verdict = ("✓ EXCELLENT" if ratio < 1.05 else
                   "~ ACCEPTABLE" if ratio < 1.20 else
                   "✗ LIMITED")
        logger.section("TSTR — Train-on-Synthetic, Test-on-Real")
        logger.write(f"  AR lags        : {ar_lags}")
        logger.write(f"  TRTR MSE       : {trtr:.8f}  (real train → real test)")
        logger.write(f"  TSTR MSE       : {tstr:.8f}  (synth train → real test)")
        logger.write(f"  Ratio          : {ratio:.4f}  {verdict}")
        logger.write(f"  n_real_train   : {result['n_real_train']}")
        logger.write(f"  n_real_test    : {result['n_real_test']}")
        logger.write(f"  n_synth_train  : {result['n_synth']}")

    return result


def diagnose(
    real: np.ndarray,
    synthetic: np.ndarray,
    figsize: Tuple[int, int] = (16, 14),
    max_lag: int = 20,
    n_sample_paths: int = 5,
    title: str = "SBBTS Diagnostic Report",
    logger=None,
):
    """
    Comprehensive diagnostic figure: paths, distribution, ACF, correlations, risk.

    Args:
        real: Real data, shape (N, T, d)
        synthetic: Synthetic data, same shape
        figsize: Figure size
        max_lag: Max ACF lag
        n_sample_paths: Paths to display in trajectory panel
        title: Figure title
        logger: optional SBBTSLogger

    Returns:
        matplotlib.Figure
    """
    _require_matplotlib()

    real      = np.array(real)
    synthetic = np.array(synthetic)
    is_multivariate = real.ndim == 3 and real.shape[-1] > 1

    if logger is not None:
        logger.section(f"Diagnostic Report — {title}")
        logger.write(f"  real shape  : {real.shape}")
        logger.write(f"  synth shape : {synthetic.shape}")

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    plot_sample_paths(real, synthetic, n_sample_paths, axes=[ax0, ax1], logger=logger)

    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    plot_marginal_comparison(real, synthetic, ax=ax2, logger=logger)

    real_diff  = np.diff(real,      axis=1) if real.ndim      == 3 else np.diff(real,      axis=-1)
    synth_diff = np.diff(synthetic, axis=1) if synthetic.ndim == 3 else np.diff(synthetic, axis=-1)
    plot_acf_comparison(real_diff, synth_diff, max_lag, ax=ax3,
                        title="ACF of Returns", logger=logger)

    if is_multivariate:
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        plot_correlation_comparison(real, synthetic, axes=[ax4, ax5], logger=logger)
    else:
        ax4 = fig.add_subplot(gs[2, :])
        plot_risk_metrics(real, synthetic, ax=ax4, logger=logger)

    return fig


def full_diagnose(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_clusters: int = 3,
    max_lag: int = 20,
    roll: int = 21,
    n_sample_paths: int = 5,
    real_1d: np.ndarray = None,
    synth_1d: np.ndarray = None,
    figsize: Tuple[int, int] = (18, 36),
    title: str = "SBBTS Full Diagnostic Report",
    logger=None,
):
    """
    Complete diagnostic figure with all panels.

    Args:
        real: Real data, shape (N, T, d)
        synthetic: Synthetic data, shape (M, T, d)
        n_clusters: K-means clusters for regime diagnostics
        max_lag: Max ACF lag
        roll: Rolling-vol window size
        n_sample_paths: Paths shown in trajectory panel
        real_1d: Optional raw 1-D return series for rolling vol (avoids window-overlap bias)
        synth_1d: Optional raw 1-D synthetic return series for rolling vol
        figsize: Overall figure size
        title: Figure title
        logger: optional SBBTSLogger

    Returns:
        matplotlib.Figure
    """
    _require_matplotlib()

    real      = np.array(real)
    synthetic = np.array(synthetic)

    if logger is not None:
        logger.section(f"Full Diagnostic Report — {title}")
        logger.write(f"  real shape  : {real.shape}")
        logger.write(f"  synth shape : {synthetic.shape}")
        logger.write(f"  n_clusters  : {n_clusters}   max_lag : {max_lag}   roll : {roll}")

    n_fixed_rows = 6 + n_clusters
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    gs  = gridspec.GridSpec(n_fixed_rows, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── Row 0: sample paths ──────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    plot_sample_paths(real, synthetic, n_sample_paths, axes=[ax00, ax01], logger=logger)

    # ── Row 1: distribution + ACF of returns ────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    r_flat = real.flatten()
    s_flat = synthetic.flatten()
    x_lo   = np.percentile(r_flat, 0.5)
    x_hi   = np.percentile(r_flat, 99.5)
    bins   = np.linspace(x_lo, x_hi, 60)
    ax10.hist(r_flat, bins=bins, density=True, alpha=0.55, label="Real",      color=REAL_C)
    ax10.hist(s_flat, bins=bins, density=True, alpha=0.55, label="Synthetic", color=SYNTH_C)
    ax10.set_xlim(x_lo, x_hi)
    ax10.set_title("Return distribution")
    ax10.set_xlabel("log return")
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    plot_marginal_comparison(real, synthetic, ax=ax10, logger=None)   # already plotted above
    plot_acf_comparison(real, synthetic, max_lag, ax=ax11,
                        title="ACF of returns", logger=logger)

    # ── Row 2: vol clustering (|r| and r²) ──────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    plot_acf_vol(real, synthetic, max_lag, axes=[ax20, ax21], logger=logger)

    # ── Row 3: QQ-plot + risk metrics ───────────────────────────────────
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    plot_qq(real, synthetic, ax=ax30, logger=logger)
    plot_risk_metrics(real, synthetic, ax=ax31, logger=logger)

    # ── Row 4-5: T×T lag-correlation heatmaps ───────────────────────────
    ax40 = fig.add_subplot(gs[4, 0])
    ax41 = fig.add_subplot(gs[4, 1])
    ax42 = fig.add_subplot(gs[5, :])
    plot_lag_corr_matrix(real, synthetic, axes=[ax40, ax41, ax42], logger=logger)

    # ── Rows 6+: per-regime diagnostics ─────────────────────────────────
    cluster_axes = np.array([
        [fig.add_subplot(gs[6 + k, 0]), fig.add_subplot(gs[6 + k, 1])]
        for k in range(n_clusters)
    ])
    plot_cluster_diagnostics(real, synthetic, n_clusters, axes=cluster_axes, logger=logger)

    # ── Rolling vol and leverage (1-D series) ────────────────────────────
    rv_real  = real_1d  if real_1d  is not None else real.flatten()
    rv_synth = synth_1d if synth_1d is not None else synthetic.flatten()
    plot_rolling_vol(rv_real, rv_synth, roll=roll, logger=logger, axes=None)
    plt.close("all")   # rolling-vol creates its own figure; close since not embedded in gs

    plot_leverage_effect(rv_real, rv_synth, max_lag=min(10, max_lag), logger=logger, axes=None)
    plt.close("all")

    plot_signature_moments(real, synthetic, depth=2, logger=logger, axes=None)
    plt.close("all")

    tstr_score(real, synthetic, ar_lags=5, logger=logger)

    plt.tight_layout()
    return fig
