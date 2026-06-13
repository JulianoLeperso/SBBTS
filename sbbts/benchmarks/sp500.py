"""
S&P 500 benchmark preprocessing for SBBTS.

Implements the feature engineering from Appendix C.2.3:
- Return-based features (cumulative, z-score)
- Volatility features
- Market-wide features

And the data augmentation pipeline from Section 5.2.
"""

from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor


def compute_cumulative_return(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute cumulative return over horizon.

    From Appendix C.2.3: feature.cum_ret_h1
        cum_ret^{(h1)}_{t,i} = Σ_{k=0}^{h1-1} R_{t-k,i}

    Args:
        returns: Return series, shape (T, d)
        horizon: Lookback horizon h1

    Returns:
        Cumulative returns, shape (T, d)
    """
    T, d = returns.shape
    cum_ret = np.zeros((T, d))

    for t in range(T):
        start = max(0, t - horizon + 1)
        cum_ret[t] = returns[start:t+1].sum(axis=0)

    return cum_ret


def compute_volatility(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute rolling volatility.

    From Appendix C.2.3: feature.vol_h1
        vol^{(h1)}_{t,i} = sqrt(1/(h1-1) * Σ(R_{t-k,i} - R̄)²)

    Args:
        returns: Return series, shape (T, d)
        horizon: Lookback horizon

    Returns:
        Volatility, shape (T, d)
    """
    T, d = returns.shape
    vol = np.zeros((T, d))

    for t in range(horizon - 1, T):
        window = returns[t - horizon + 1:t + 1]
        vol[t] = np.std(window, axis=0, ddof=1)

    return vol


def compute_zscore(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute z-score of lag-1 return.

    From Appendix C.2.3: feature.ret_t-1_zscore_h
        z^{(h)}_{t,i} = (R_{t-1,i} - μ^{(h)}_{t,i}) / σ^{(h)}_{t,i}

    Args:
        returns: Return series, shape (T, d)
        horizon: Lookback horizon

    Returns:
        Z-scores, shape (T, d)
    """
    T, d = returns.shape
    zscore = np.zeros((T, d))

    for t in range(horizon, T):
        window = returns[t - horizon:t]
        mu = np.mean(window, axis=0)
        sigma = np.std(window, axis=0, ddof=1) + 1e-8
        zscore[t] = (returns[t - 1] - mu) / sigma

    return zscore


def compute_market_return(returns: np.ndarray) -> np.ndarray:
    """
    Compute market-wide average return.

    From Appendix C.2.3: feature.return_t-1_market
        R̃_{t-1} = (1/d) * Σ R_{t-1,i}

    Args:
        returns: Return series, shape (T, d)

    Returns:
        Market return, shape (T,)
    """
    return np.mean(returns, axis=1)


def compute_market_volatility(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute market-wide volatility.

    From Appendix C.2.3: feature.mkt_vol_h

    Args:
        returns: Return series, shape (T, d)
        horizon: Lookback horizon

    Returns:
        Market volatility, shape (T,)
    """
    mkt_returns = compute_market_return(returns)
    T = len(mkt_returns)
    vol = np.zeros(T)

    for t in range(horizon - 1, T):
        window = mkt_returns[t - horizon + 1:t + 1]
        vol[t] = np.std(window, ddof=1)

    return vol


def compute_market_cumret(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute market cumulative return.

    From Appendix C.2.3: feature.mkt_cumret_h

    Args:
        returns: Return series, shape (T, d)
        horizon: Lookback horizon

    Returns:
        Market cumulative return, shape (T,)
    """
    mkt_returns = compute_market_return(returns)
    T = len(mkt_returns)
    cumret = np.zeros(T)

    for t in range(T):
        start = max(0, t - horizon + 1)
        cumret[t] = mkt_returns[start:t+1].sum()

    return cumret


def engineer_features(
    returns: np.ndarray,
    cum_ret_horizons: List[int] = [5, 10, 21, 63, 126, 252],
    zscore_horizons: List[int] = [3, 5, 10, 21],
    vol_horizons: List[int] = [5, 10, 21, 63, 126, 252],
    mkt_horizons: List[int] = [5, 10, 21],
) -> Tuple[np.ndarray, List[str]]:
    """
    Engineer all features as in Appendix C.2.3.

    Args:
        returns: Return matrix R ∈ (T, d)
        cum_ret_horizons: Horizons for cumulative return features
        zscore_horizons: Horizons for z-score features
        vol_horizons: Horizons for volatility features
        mkt_horizons: Horizons for market features

    Returns:
        Tuple of:
            - Feature matrix, shape (T, n_features)
            - Feature names
    """
    T, d = returns.shape
    features = []
    feature_names = []

    lag1_mkt = np.zeros(T)
    lag1_mkt[1:] = compute_market_return(returns)[:-1]
    features.append(lag1_mkt.reshape(-1, 1))
    feature_names.append("return_t-1_market")

    for h in cum_ret_horizons:
        feat = compute_cumulative_return(returns, h)
        features.append(feat)
        feature_names.extend([f"cum_ret_{h}_inst{i}" for i in range(d)])

    for h in vol_horizons:
        feat = compute_volatility(returns, h)
        features.append(feat)
        feature_names.extend([f"vol_{h}_inst{i}" for i in range(d)])

    for h in zscore_horizons:
        feat = compute_zscore(returns, h)
        features.append(feat)
        feature_names.extend([f"zscore_{h}_inst{i}" for i in range(d)])

    for h in mkt_horizons:
        feat = compute_market_cumret(returns, h)
        features.append(feat.reshape(-1, 1))
        feature_names.append(f"mkt_cumret_{h}")

        feat = compute_market_volatility(returns, h)
        features.append(feat.reshape(-1, 1))
        feature_names.append(f"mkt_vol_{h}")

    X = np.hstack(features)
    return X, feature_names


class SP500DataProcessor:
    """
    Data processor for S&P 500 experiments.

    Handles the full pipeline from Section 5.2:
    1. Load/generate return data
    2. Feature engineering (Appendix C.2.3)
    3. Train/val/test split
    4. Data augmentation with SBBTS
    """

    def __init__(
        self,
        context_length: int = 22,
        max_lookback: int = 252,
    ):
        """
        Args:
            context_length: TabICL context window (Section C.2.2: 22 days)
            max_lookback: Maximum lookback for features (252 days = 1 year)
        """
        self.context_length = context_length
        self.max_lookback = max_lookback
        self._features = None
        self._feature_names = None

    def process_returns(
        self,
        returns: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process returns into features.

        Args:
            returns: Raw returns, shape (T, d)

        Returns:
            Features and names
        """
        features, names = engineer_features(returns)
        self._features = features
        self._feature_names = names
        return features, names

    def create_episodes(
        self,
        returns: np.ndarray,
        features: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create forecasting episodes for TabICL.

        From Appendix C.2.2: An episode is defined by a context window
        of n_context days, predicting the next day's return sign.

        Args:
            returns: Return matrix, shape (T, d)
            features: Feature matrix, shape (T, n_features)

        Returns:
            Tuple of:
                - X: Context features, shape (n_episodes, context_length, n_features)
                - y: Target signs, shape (n_episodes, d)
        """
        if features is None:
            features, _ = self.process_returns(returns)

        T = returns.shape[0]
        n_episodes = T - self.context_length - self.max_lookback

        X = []
        y = []

        for start in range(self.max_lookback, T - self.context_length):
            end = start + self.context_length
            X.append(features[start:end])
            y.append((returns[end] > 0).astype(float))

        return np.array(X), np.array(y)

    def train_val_test_split(
        self,
        returns: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train/val/test as in Section 5.2.3.

        Args:
            returns: Full return matrix
            train_end: Index where training ends
            val_end: Index where validation ends

        Returns:
            Dictionary with train/val/test splits
        """
        return {
            "train": returns[:train_end],
            "val": returns[train_end:val_end],
            "test": returns[val_end:],
        }


def compute_pnl(
    predictions: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute daily PnL from predictions.

    From Section 5.2.4:
        w_t = 2 * p̂_t - 1 ∈ [-1, 1]
        PnL_t = (1/d) * w_t^T * R_t

    Args:
        predictions: Predicted probabilities p̂_t ∈ [0, 1], shape (T, d)
        returns: True returns R_t, shape (T, d)

    Returns:
        Daily PnL, shape (T,)
    """
    positions = 2 * predictions - 1
    pnl = np.mean(positions * returns, axis=1)
    return pnl


def compute_sharpe_from_pnl(
    pnl: np.ndarray,
    annualization: int = 252,
) -> float:
    """
    Compute Sharpe ratio from PnL series.

    From Section 5.2.4:
        Sharpe = (mean PnL / std PnL) * sqrt(252)

    Args:
        pnl: Daily PnL series
        annualization: Trading days per year

    Returns:
        Annualized Sharpe ratio
    """
    if np.std(pnl) == 0:
        return 0.0
    return np.mean(pnl) / np.std(pnl) * np.sqrt(annualization)


def generate_synthetic_sp500(
    n_instruments: int = 433,
    n_days: int = 2263,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic S&P 500-like returns for testing.

    Args:
        n_instruments: Number of instruments (S&P 500: 433)
        n_days: Number of trading days
        seed: Random seed

    Returns:
        Returns matrix, shape (n_days, n_instruments)
    """
    np.random.seed(seed)

    factor_returns = np.random.randn(n_days, 3) * 0.01

    loadings = np.random.randn(n_instruments, 3) * 0.5

    idio_returns = np.random.randn(n_days, n_instruments) * 0.02

    returns = factor_returns @ loadings.T + idio_returns

    return returns
