"""
Heston parameter recovery experiment.

Reproduces Figure 2 from the SBBTS paper:
1. Generate Heston trajectories with heterogeneous parameters
2. Train SBBTS on the dataset
3. Generate synthetic trajectories
4. Estimate parameters on synthetic data via MLE
5. Compare parameter distributions

Usage:
    python -m sbbts.examples.heston_recovery
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from sbbts import SBBTS
from sbbts.benchmarks.heston import (
    generate_heston_dataset,
    estimate_heston_mle,
    evaluate_parameter_recovery,
    HestonParams,
)


def run_heston_recovery(
    n_train: int = 5000,
    n_synth: int = 5000,
    trajectory_length: int = 252,
    beta: float = 300.0,
    n_steps: int = 5,
    n_epochs: int = 100,
    batch_size: int = 128,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run the Heston parameter recovery experiment.

    Args:
        n_train: Number of training trajectories (Section 5.1: 5000)
        n_synth: Number of synthetic trajectories to generate (Section 5.1: 5000)
        trajectory_length: Days per trajectory (Section 5.1: 252)
        beta: SBBTS β parameter
        n_steps: SBBTS K iterations
        n_epochs: Training epochs
        batch_size: Batch size
        verbose: Print progress
        seed: Random seed

    Returns:
        Dictionary with results
    """
    if verbose:
        print("=" * 60)
        print("SBBTS Heston Parameter Recovery Experiment")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Training trajectories: {n_train}")
        print(f"  Synthetic trajectories: {n_synth}")
        print(f"  Trajectory length: {trajectory_length}")
        print(f"  β = {beta}, K = {n_steps}, epochs = {n_epochs}")

    if verbose:
        print("\n1. Generating Heston training data...")
    X_train, train_params = generate_heston_dataset(
        n_trajectories=n_train,
        trajectory_length=trajectory_length,
        heterogeneous=True,
        seed=seed,
    )
    if verbose:
        print(f"   Shape: {X_train.shape}")

    if verbose:
        print("\n2. Training SBBTS model...")
    model = SBBTS(
        beta=beta,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        d_model=128,
        n_heads=16,
    )
    model.fit(X_train, verbose=verbose)

    if verbose:
        print("\n3. Generating synthetic trajectories...")
    X_synth = model.sample(n=n_synth)
    if verbose:
        print(f"   Shape: {X_synth.shape}")

    if verbose:
        print("\n4. Estimating parameters via MLE...")

    train_est_params = []
    for i in range(min(n_train, 1000)):
        params = estimate_heston_mle(X_train[i])
        train_est_params.append(params)

    synth_est_params = []
    for i in range(min(n_synth, 1000)):
        params = estimate_heston_mle(X_synth[i])
        synth_est_params.append(params)

    if verbose:
        print("\n5. Evaluating parameter recovery...")

    train_eval = evaluate_parameter_recovery(
        train_params[: len(train_est_params)], train_est_params
    )
    synth_eval = evaluate_parameter_recovery(
        train_params[: len(synth_est_params)], synth_est_params
    )

    if verbose:
        print("\nResults (Real Data MLE):")
        for param, stats in train_eval.items():
            print(
                f"  {param}: true_mean={stats['true_mean']:.3f}, "
                f"est_mean={stats['est_mean']:.3f}, rmse={stats['rmse']:.3f}"
            )

        print("\nResults (Synthetic Data MLE):")
        for param, stats in synth_eval.items():
            print(
                f"  {param}: true_mean={stats['true_mean']:.3f}, "
                f"est_mean={stats['est_mean']:.3f}, rmse={stats['rmse']:.3f}"
            )

    return {
        "model": model,
        "X_train": X_train,
        "X_synth": X_synth,
        "train_params": train_params,
        "train_est_params": train_est_params,
        "synth_est_params": synth_est_params,
        "train_eval": train_eval,
        "synth_eval": synth_eval,
    }


def plot_parameter_distributions(
    train_params: List[HestonParams],
    train_est_params: List[HestonParams],
    synth_est_params: List[HestonParams],
    save_path: str = None,
):
    """
    Plot parameter distributions as in Figure 2.

    Args:
        train_params: True parameters
        train_est_params: MLE estimates on real data
        synth_est_params: MLE estimates on synthetic data
        save_path: Path to save figure
    """
    param_names = ["kappa", "theta", "xi", "rho", "r"]
    param_labels = [r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$", r"$r$"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i, (name, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]

        true_vals = [getattr(p, name) for p in train_params[: len(train_est_params)]]
        train_vals = [getattr(p, name) for p in train_est_params]
        synth_vals = [getattr(p, name) for p in synth_est_params]

        ax.hist(true_vals, bins=30, alpha=0.5, density=True, label="True", color="blue")
        ax.hist(train_vals, bins=30, alpha=0.5, density=True, label="Real MLE", color="orange")
        ax.hist(synth_vals, bins=30, alpha=0.5, density=True, label="SBBTS MLE", color="green")

        ax.set_xlabel(label)
        ax.set_ylabel("Density" if i == 0 else "")
        if i == 0:
            ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    results = run_heston_recovery(
        n_train=1000,
        n_synth=1000,
        n_epochs=50,
        verbose=True,
    )

    plot_parameter_distributions(
        results["train_params"][: len(results["train_est_params"])],
        results["train_est_params"],
        results["synth_est_params"],
    )
