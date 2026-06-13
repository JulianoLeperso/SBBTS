"""
Data augmentation example for time series forecasting.

Demonstrates the SBBTS augmentation pipeline from Section 5.2:
1. Load/generate return data
2. Apply PCA + k-means dimensionality reduction
3. Train SBBTS on reduced factors
4. Generate synthetic data (×200 augmentation factor)
5. Evaluate downstream forecasting improvement

Usage:
    python -m sbbts.examples.augmentation
"""

import numpy as np
from typing import Dict, Tuple

from sbbts import SBBTS
from sbbts.utils.dim_reduction import PCAKMeansReducer
from sbbts.utils.metrics import compute_all_risk_metrics, sharpe_ratio
from sbbts.benchmarks.sp500 import (
    generate_synthetic_sp500,
    compute_pnl,
    compute_sharpe_from_pnl,
)


def run_augmentation_experiment(
    n_instruments: int = 50,
    n_train_days: int = 500,
    n_test_days: int = 100,
    augmentation_factor: int = 10,
    n_pca_components: int = 8,
    n_clusters: int = 3,
    beta: float = 50.0,
    n_steps: int = 3,
    n_epochs: int = 50,
    batch_size: int = 64,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run data augmentation experiment.

    Args:
        n_instruments: Number of instruments
        n_train_days: Training period length
        n_test_days: Test period length
        augmentation_factor: Synthetic data multiplier (Table 1: 200)
        n_pca_components: PCA components (Appendix C.2.4: 16)
        n_clusters: K-means clusters (Appendix C.2.4: 3)
        beta: SBBTS β parameter
        n_steps: SBBTS K iterations
        n_epochs: Training epochs per iteration
        batch_size: Training batch size
        verbose: Print progress
        seed: Random seed

    Returns:
        Dictionary with experiment results
    """
    np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("SBBTS Data Augmentation Experiment")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Instruments: {n_instruments}")
        print(f"  Train days: {n_train_days}, Test days: {n_test_days}")
        print(f"  Augmentation factor: {augmentation_factor}×")
        print(f"  PCA components: {n_pca_components}, Clusters: {n_clusters}")

    if verbose:
        print("\n1. Generating synthetic market data...")
    returns = generate_synthetic_sp500(
        n_instruments=n_instruments,
        n_days=n_train_days + n_test_days,
        seed=seed,
    )
    train_returns = returns[:n_train_days]
    test_returns = returns[n_train_days:]
    if verbose:
        print(f"   Train shape: {train_returns.shape}")
        print(f"   Test shape: {test_returns.shape}")

    if verbose:
        print("\n2. Applying PCA + K-means reduction...")
    reducer = PCAKMeansReducer(
        n_components=n_pca_components,
        n_clusters=n_clusters,
    )

    window_size = 20
    n_windows = n_train_days - window_size + 1
    X_train_windows = np.array([train_returns[i : i + window_size] for i in range(n_windows)])

    X_reduced = reducer.fit_transform(X_train_windows)
    if verbose:
        print(f"   Reduced shape: {X_reduced.shape}")
        print(f"   Explained variance: {reducer.explained_variance_ratio.sum():.2%}")

    if verbose:
        print("\n3. Training SBBTS on reduced factors...")
    model = SBBTS(
        beta=beta,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        d_model=64,
        n_heads=8,
    )
    model.fit(X_reduced, verbose=verbose)

    if verbose:
        print("\n4. Generating synthetic data...")
    n_synth = n_windows * augmentation_factor
    X_synth_reduced = model.sample(n=n_synth)
    X_synth = reducer.inverse_transform(X_synth_reduced)
    if verbose:
        print(f"   Synthetic shape: {X_synth.shape}")

    if verbose:
        print("\n5. Computing metrics...")

    real_returns_flat = train_returns.flatten()
    synth_returns_flat = X_synth.reshape(-1, n_instruments).flatten()

    real_metrics = compute_all_risk_metrics(real_returns_flat)
    synth_metrics = compute_all_risk_metrics(synth_returns_flat)

    if verbose:
        print("\nReal data metrics:")
        for k, v in real_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nSynthetic data metrics:")
        for k, v in synth_metrics.items():
            print(f"  {k}: {v:.4f}")

    if verbose:
        print("\n6. Evaluating with simple momentum strategy...")

    def simple_momentum_predict(returns_history, lookback=5):
        """Simple momentum: predict direction = sign of recent avg return."""
        n_days, n_inst = returns_history.shape
        preds = np.zeros((n_days - lookback, n_inst))
        for t in range(lookback, n_days):
            preds[t - lookback] = (returns_history[t - lookback : t].mean(axis=0) > 0).astype(float)
        return preds

    preds_real = simple_momentum_predict(train_returns)
    pnl_real = compute_pnl(preds_real, train_returns[5:])
    sharpe_real = compute_sharpe_from_pnl(pnl_real)

    augmented_returns = np.vstack([train_returns, X_synth.mean(axis=1)])
    preds_aug = simple_momentum_predict(augmented_returns)
    pnl_aug = compute_pnl(preds_aug[: len(pnl_real)], train_returns[5:])
    sharpe_aug = compute_sharpe_from_pnl(pnl_aug)

    if verbose:
        print(f"\n  Real-only training Sharpe: {sharpe_real:.3f}")
        print(f"  Augmented training Sharpe: {sharpe_aug:.3f}")

    results = {
        "model": model,
        "reducer": reducer,
        "train_returns": train_returns,
        "test_returns": test_returns,
        "X_synth": X_synth,
        "real_metrics": real_metrics,
        "synth_metrics": synth_metrics,
        "sharpe_real": sharpe_real,
        "sharpe_aug": sharpe_aug,
    }

    return results


def compare_autocorrelation(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    max_lag: int = 20,
):
    """
    Compare autocorrelation of real vs synthetic data.

    As in Figure 5 of the paper.

    Args:
        real_data: Real returns
        synth_data: Synthetic returns
        max_lag: Maximum lag for ACF
    """
    import matplotlib.pyplot as plt
    from sbbts.utils.metrics import autocorrelation

    real_flat = real_data.flatten()
    synth_flat = synth_data.flatten()

    acf_real = autocorrelation(real_flat, max_lag)
    acf_synth = autocorrelation(synth_flat, max_lag)

    acf_real_sq = autocorrelation(real_flat**2, max_lag)
    acf_synth_sq = autocorrelation(synth_flat**2, max_lag)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(max_lag + 1), acf_real, alpha=0.5, label="Real", width=0.4)
    axes[0].bar(np.arange(max_lag + 1) + 0.4, acf_synth, alpha=0.5, label="Synthetic", width=0.4)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[0].set_title("Autocorrelation of Returns")
    axes[0].legend()

    axes[1].bar(range(max_lag + 1), acf_real_sq, alpha=0.5, label="Real", width=0.4)
    axes[1].bar(np.arange(max_lag + 1) + 0.4, acf_synth_sq, alpha=0.5, label="Synthetic", width=0.4)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("Autocorrelation of Squared Returns")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_augmentation_experiment(
        n_instruments=30,
        n_train_days=200,
        n_test_days=50,
        augmentation_factor=5,
        n_pca_components=5,
        n_epochs=30,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
