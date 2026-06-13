"""
Evaluation metrics for SBBTS.

Implements metrics from Table 4 and Section 5.2.4:
- VaR (Value at Risk) at 95% and 99%
- ES (Expected Shortfall) at 95% and 99%
- Sharpe ratio (annualized)
- Autocorrelation functions
"""

from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor


def to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_returns(
    prices: Union[np.ndarray, Tensor],
    log_returns: bool = True,
) -> np.ndarray:
    """
    Compute returns from price series.

    Args:
        prices: Price series, shape (..., T) or (..., T, d)
        log_returns: Use log returns (True) or simple returns (False)

    Returns:
        Returns, shape (..., T-1) or (..., T-1, d)
    """
    prices = to_numpy(prices)

    if log_returns:
        returns = np.diff(np.log(prices), axis=-2 if prices.ndim > 1 else -1)
    else:
        returns = np.diff(prices, axis=-2 if prices.ndim > 1 else -1) / prices[..., :-1, :] if prices.ndim > 2 else np.diff(prices) / prices[:-1]

    return returns


def var(
    returns: Union[np.ndarray, Tensor],
    confidence: float = 0.95,
) -> float:
    """
    Compute Value at Risk.

    VaR_α is the α-quantile of the loss distribution.
    Table 4: VaR99%, VaR95%

    Args:
        returns: Return series, shape (n_samples,) or flattened
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR at the specified confidence level (positive = loss)
    """
    returns = to_numpy(returns).flatten()
    return -np.percentile(returns, (1 - confidence) * 100)


def expected_shortfall(
    returns: Union[np.ndarray, Tensor],
    confidence: float = 0.95,
) -> float:
    """
    Compute Expected Shortfall (Conditional VaR).

    ES_α is the expected loss given that loss exceeds VaR_α.
    Table 4: ES99%, ES95%

    Args:
        returns: Return series, shape (n_samples,) or flattened
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        ES at the specified confidence level (positive = loss)
    """
    returns = to_numpy(returns).flatten()
    var_value = var(returns, confidence)
    tail_returns = returns[returns <= -var_value]
    if len(tail_returns) == 0:
        return var_value
    return -np.mean(tail_returns)


def sharpe_ratio(
    returns: Union[np.ndarray, Tensor],
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Sharpe = (mean_return - rf) / std_return * sqrt(annualization_factor)

    From Section 5.2.4 and Table 1.

    Args:
        returns: Daily returns, shape (n_samples,) or (n_samples, n_instruments)
        annualization_factor: Trading days per year (default 252)
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Annualized Sharpe ratio
    """
    returns = to_numpy(returns)

    if returns.ndim > 1:
        returns = returns.mean(axis=-1)

    daily_rf = risk_free_rate / annualization_factor

    mean_return = np.mean(returns) - daily_rf
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    return mean_return / std_return * np.sqrt(annualization_factor)


def autocorrelation(
    series: Union[np.ndarray, Tensor],
    max_lag: int = 20,
) -> np.ndarray:
    """
    Compute autocorrelation function.

    Used in Figure 5 for comparing real vs synthetic data.

    Args:
        series: Time series, shape (T,) or (T, d)
        max_lag: Maximum lag to compute

    Returns:
        Autocorrelation values for lags 0 to max_lag
    """
    series = to_numpy(series)

    if series.ndim > 1:
        series = series.mean(axis=-1)

    n = len(series)
    mean = np.mean(series)
    var = np.var(series)

    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
            acf[lag] = cov / var

    return acf


def compute_all_risk_metrics(
    returns: Union[np.ndarray, Tensor],
    annualization_factor: int = 252,
) -> dict:
    """
    Compute all risk metrics from Table 4.

    Args:
        returns: Return series
        annualization_factor: Trading days per year

    Returns:
        Dictionary with all metrics
    """
    returns = to_numpy(returns).flatten()

    return {
        "var_95": var(returns, 0.95),
        "var_99": var(returns, 0.99),
        "es_95": expected_shortfall(returns, 0.95),
        "es_99": expected_shortfall(returns, 0.99),
        "ann_return": np.mean(returns) * annualization_factor,
        "ann_std": np.std(returns, ddof=1) * np.sqrt(annualization_factor),
        "sharpe": sharpe_ratio(returns, annualization_factor),
    }


def compare_distributions(
    real: Union[np.ndarray, Tensor],
    synthetic: Union[np.ndarray, Tensor],
) -> dict:
    """
    Compare statistical properties of real vs synthetic data.

    Used for validation as in Appendix C.2.1.

    Args:
        real: Real data
        synthetic: Synthetic data

    Returns:
        Dictionary of comparison metrics
    """
    real = to_numpy(real).flatten()
    synthetic = to_numpy(synthetic).flatten()

    real_metrics = compute_all_risk_metrics(real)
    synth_metrics = compute_all_risk_metrics(synthetic)

    comparison = {}
    for key in real_metrics:
        comparison[f"real_{key}"] = real_metrics[key]
        comparison[f"synth_{key}"] = synth_metrics[key]
        if real_metrics[key] != 0:
            comparison[f"diff_{key}_pct"] = (
                (synth_metrics[key] - real_metrics[key]) / abs(real_metrics[key]) * 100
            )

    return comparison


def log_ret_to_prices(
    X: Union[np.ndarray, Tensor],
    S0: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    Convert log return windows to price paths.

    Args:
        X:  Log return trajectories, shape (N, T, d)
        S0: Initial price — scalar or shape (d,)

    Returns:
        Price paths of shape (N, T+1, d), where path[:, 0, :] = S0
    """
    X = to_numpy(X)
    zeros = np.zeros((*X.shape[:-2], 1, X.shape[-1]))
    cum = np.concatenate([zeros, np.cumsum(X, axis=-2)], axis=-2)
    return np.asarray(S0) * np.exp(cum)


def compute_metrics(
    X_real: Union[np.ndarray, Tensor],
    X_synth: Union[np.ndarray, Tensor],
    annualization_factor: int = 252,
    acf_lags: int = 5,
) -> dict:
    """
    Compare stylized facts between real and synthetic windows.

    Returns a structured dict so results can be logged, asserted in CI,
    or displayed without touching matplotlib.

    Args:
        X_real:  Real trajectories,      shape (N,  T, d)
        X_synth: Synthetic trajectories, shape (N2, T, d)
        annualization_factor: Trading days per year (default 252)
        acf_lags: Number of lags to sum for vol-clustering score

    Returns:
        dict with keys  <metric>_real, <metric>_synth, <metric>_ratio
        for each stylized fact, plus raw scalar values.
    """
    X_real  = to_numpy(X_real).astype(np.float64)
    X_synth = to_numpy(X_synth).astype(np.float64)

    def _flatten(X):
        return X.reshape(-1)

    def _per_window_rv(X):
        return np.sum(X ** 2, axis=1)          # (N, d) — realized variance per window

    def _acf_abs_sum(X, max_lag):
        r = _flatten(X)
        abs_r = np.abs(r)
        mean = abs_r.mean()
        var  = abs_r.var()
        if var == 0:
            return 0.0
        total = 0.0
        for lag in range(1, max_lag + 1):
            cov = np.mean((abs_r[:-lag] - mean) * (abs_r[lag:] - mean))
            total += cov / var
        return float(total)

    def _leverage(X, k=1):
        r = _flatten(X)
        if len(r) <= k:
            return 0.0
        return float(np.corrcoef(r[:-k], r[k:] ** 2)[0, 1])

    def _rolling_vol(X, window=20):
        r = _flatten(X)
        if len(r) < window:
            return 0.0, 0.0
        vols = [r[i:i+window].std(ddof=1) * np.sqrt(annualization_factor)
                for i in range(len(r) - window + 1)]
        return float(np.mean(vols)), float(np.std(vols, ddof=1))

    r_real  = _flatten(X_real)
    r_synth = _flatten(X_synth)

    rv_real  = _per_window_rv(X_real).flatten()
    rv_synth = _per_window_rv(X_synth).flatten()

    ann_std_real  = float(r_real.std(ddof=1)  * np.sqrt(annualization_factor))
    ann_std_synth = float(r_synth.std(ddof=1) * np.sqrt(annualization_factor))

    from scipy.stats import kurtosis as _kurt
    kurt_real  = float(_kurt(r_real,  fisher=True))
    kurt_synth = float(_kurt(r_synth, fisher=True))

    acf_real  = _acf_abs_sum(X_real,  acf_lags)
    acf_synth = _acf_abs_sum(X_synth, acf_lags)

    lev_real  = _leverage(X_real)
    lev_synth = _leverage(X_synth)

    rvol_mean_real,  rvol_std_real  = _rolling_vol(X_real)
    rvol_mean_synth, rvol_std_synth = _rolling_vol(X_synth)

    def _ratio(s, r):
        return float(s / r) if r != 0 else float("nan")

    return {
        # Annualised volatility
        "ann_std_real":          ann_std_real,
        "ann_std_synth":         ann_std_synth,
        "ann_std_ratio":         _ratio(ann_std_synth, ann_std_real),
        # Realised variance (mean and cross-window std)
        "rv_mean_real":          float(rv_real.mean()),
        "rv_mean_synth":         float(rv_synth.mean()),
        "rv_mean_ratio":         _ratio(rv_synth.mean(), rv_real.mean()),
        "rv_std_real":           float(rv_real.std(ddof=1)),
        "rv_std_synth":          float(rv_synth.std(ddof=1)),
        "rv_std_ratio":          _ratio(rv_synth.std(ddof=1), rv_real.std(ddof=1)),
        # Tail heaviness
        "kurtosis_real":         kurt_real,
        "kurtosis_synth":        kurt_synth,
        "kurtosis_ratio":        _ratio(kurt_synth, kurt_real),
        # Volatility clustering (sum of ACF of |r| at lags 1..acf_lags)
        "acf_abs_sum_real":      acf_real,
        "acf_abs_sum_synth":     acf_synth,
        "acf_abs_sum_ratio":     _ratio(acf_synth, acf_real),
        # Leverage effect (k=1)
        "leverage_k1_real":      lev_real,
        "leverage_k1_synth":     lev_synth,
        # Rolling volatility
        "rolling_vol_mean_real":  rvol_mean_real,
        "rolling_vol_mean_synth": rvol_mean_synth,
        "rolling_vol_mean_ratio": _ratio(rvol_mean_synth, rvol_mean_real),
        "rolling_vol_std_real":   rvol_std_real,
        "rolling_vol_std_synth":  rvol_std_synth,
    }


def compute_tstr(
    X_real: Union[np.ndarray, Tensor],
    X_synth: Union[np.ndarray, Tensor],
    ar_order: int = 5,
    test_fraction: float = 0.2,
) -> dict:
    """
    Train-on-Synthetic, Test-on-Real (TSTR) evaluation.

    Fits a linear AR(ar_order) model on synthetic data, evaluates its MSE
    on a held-out real test set, and compares to TRTR (same model fit on real
    train data).  Ratio close to 1.0 means synthetic is a near-perfect
    substitute for real data for this forecasting task.

    Args:
        X_real:       Real trajectories,      shape (N,  T, d)
        X_synth:      Synthetic trajectories, shape (N2, T, d)
        ar_order:     Autoregressive order p
        test_fraction: Fraction of X_real held out for testing

    Returns:
        dict with keys trtr_mse, tstr_mse, ratio,
                       n_real_train, n_real_test, n_synth_train
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        raise ImportError("compute_tstr requires scikit-learn: pip install scikit-learn")

    X_real  = to_numpy(X_real).astype(np.float32)
    X_synth = to_numpy(X_synth).astype(np.float32)
    N, T, d = X_real.shape
    p = ar_order

    def _make_ar_dataset(X):
        N_, T_, d_ = X.shape
        feats, tgts = [], []
        for t in range(p, T_):
            feats.append(X[:, t - p:t, :].reshape(N_, p * d_))
            tgts.append(X[:, t, :])
        return (np.concatenate(feats, axis=0),
                np.concatenate(tgts,  axis=0))

    n_test  = max(1, int(N * test_fraction))
    n_train = N - n_test
    X_real_train, X_real_test = X_real[:n_train], X_real[n_train:]

    feat_train_real, tgt_train_real = _make_ar_dataset(X_real_train)
    feat_test,       tgt_test       = _make_ar_dataset(X_real_test)
    feat_synth,      tgt_synth      = _make_ar_dataset(X_synth)

    model_real  = Ridge(alpha=1e-3).fit(feat_train_real, tgt_train_real)
    model_synth = Ridge(alpha=1e-3).fit(feat_synth,      tgt_synth)

    trtr_mse = float(np.mean((model_real.predict(feat_test)  - tgt_test) ** 2))
    tstr_mse = float(np.mean((model_synth.predict(feat_test) - tgt_test) ** 2))

    return {
        "trtr_mse":      trtr_mse,
        "tstr_mse":      tstr_mse,
        "ratio":         tstr_mse / trtr_mse if trtr_mse > 0 else float("nan"),
        "n_real_train":  n_train,
        "n_real_test":   n_test,
        "n_synth_train": len(X_synth),
    }


class MetricsTracker:
    """
    Track and aggregate metrics during evaluation.
    """

    def __init__(self):
        self.metrics = {}

    def add(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def mean(self, name: str) -> float:
        """Get mean of metric."""
        if name not in self.metrics:
            return 0.0
        return np.mean(self.metrics[name])

    def std(self, name: str) -> float:
        """Get std of metric."""
        if name not in self.metrics:
            return 0.0
        return np.std(self.metrics[name], ddof=1)

    def summary(self) -> dict:
        """Get summary of all metrics."""
        return {
            name: {"mean": self.mean(name), "std": self.std(name)}
            for name in self.metrics
        }
