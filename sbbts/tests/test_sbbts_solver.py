"""
Tests for SBBTS solver.

Validates:
1. Smoke test: fit on GBM trajectories, sample, verify shapes
2. β * Δt > 1 condition is validated
3. API consistency (sklearn-like interface)
"""

import torch
import numpy as np
import pytest
from sbbts.core.sbbts_solver import SBBTS


def generate_gbm_trajectories(
    n_samples: int,
    n_steps: int,
    d: int = 1,
    mu: float = 0.05,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    S0: float = 100.0,
) -> np.ndarray:
    """Generate Geometric Brownian Motion trajectories."""
    np.random.seed(42)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(
        n_samples, n_steps, d
    )
    log_prices = np.cumsum(log_returns, axis=1)
    log_prices = np.concatenate([np.zeros((n_samples, 1, d)), log_prices], axis=1)
    return log_prices


class TestSBBTSInit:
    """Test SBBTS initialization."""

    def test_default_init(self):
        """Should initialize with default parameters."""
        model = SBBTS()
        assert model.beta == 10.0
        assert model.n_steps == 5
        assert model.d_model == 128

    def test_custom_params(self):
        """Should accept custom parameters."""
        model = SBBTS(beta=20.0, n_steps=3, d_model=64)
        assert model.beta == 20.0
        assert model.n_steps == 3
        assert model.d_model == 64

    def test_invalid_beta_raises(self):
        """Should raise for non-positive β."""
        with pytest.raises(ValueError):
            SBBTS(beta=0.0)
        with pytest.raises(ValueError):
            SBBTS(beta=-1.0)


class TestBetaCondition:
    """Test β * Δt > 1 validation (Theorem 3.2)."""

    def test_valid_beta_accepted(self):
        """Should accept valid β values."""
        model = SBBTS(beta=300.0)
        X = generate_gbm_trajectories(n_samples=10, n_steps=252, d=2)
        model._validate_beta_for_data(n_time_points=253, T=1.0)

    def test_invalid_beta_rejected(self):
        """Should reject β that violates Theorem 3.2."""
        model = SBBTS(beta=10.0)
        with pytest.raises(ValueError, match="Theorem 3.2"):
            model._validate_beta_for_data(n_time_points=253, T=1.0)


class TestSBBTSFit:
    """Test SBBTS fitting."""

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for testing."""
        return generate_gbm_trajectories(n_samples=50, n_steps=10, d=2)

    def test_fit_returns_self(self, small_dataset):
        """fit() should return self for chaining."""
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        result = model.fit(small_dataset, verbose=False)
        assert result is model

    def test_fit_marks_fitted(self, small_dataset):
        """fit() should set _fitted flag."""
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        assert not model._fitted
        model.fit(small_dataset, verbose=False)
        assert model._fitted

    def test_fit_creates_score_network(self, small_dataset):
        """fit() should create score network."""
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        assert model.score_net is None
        model.fit(small_dataset, verbose=False)
        assert model.score_net is not None

    def test_fit_accepts_numpy(self, small_dataset):
        """Should accept numpy arrays."""
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        model.fit(small_dataset, verbose=False)
        assert model._fitted

    def test_fit_accepts_tensor(self, small_dataset):
        """Should accept PyTorch tensors."""
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        X_tensor = torch.from_numpy(small_dataset).float()
        model.fit(X_tensor, verbose=False)
        assert model._fitted


class TestSBBTSSample:
    """Test SBBTS sampling."""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing."""
        X = generate_gbm_trajectories(n_samples=50, n_steps=10, d=2)
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        model.fit(X, verbose=False)
        return model

    def test_sample_before_fit_raises(self):
        """sample() should raise if not fitted."""
        model = SBBTS()
        with pytest.raises(RuntimeError, match="fitted"):
            model.sample(n=10)

    def test_sample_shape(self, fitted_model):
        """sample() should return correct shape."""
        X_synth = fitted_model.sample(n=20)
        assert X_synth.shape == (20, 11, 2)

    def test_sample_final_only(self, fitted_model):
        """sample() can return only final state."""
        X_final = fitted_model.sample(n=20, return_full_trajectory=False)
        assert X_final.shape == (20, 2)

    def test_sample_with_init(self, fitted_model):
        """sample() should accept initial states."""
        X_init = np.random.randn(15, 2)
        X_synth = fitted_model.sample(n=15, X_init=X_init)
        assert X_synth.shape == (15, 11, 2)


class TestSBBTSAugment:
    """Test data augmentation."""

    @pytest.fixture
    def fitted_model_and_data(self):
        """Create fitted model and test data."""
        X = generate_gbm_trajectories(n_samples=30, n_steps=10, d=2)
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        model.fit(X, verbose=False)
        return model, X

    def test_augment_shape(self, fitted_model_and_data):
        """augment() should return correct shape."""
        model, X = fitted_model_and_data
        X_aug = model.augment(X, factor=2)
        assert X_aug.shape == (30 + 60, 11, 2)


class TestSBBTSSaveLoad:
    """Test model persistence."""

    @pytest.fixture
    def fitted_model(self, tmp_path):
        """Create fitted model for testing."""
        X = generate_gbm_trajectories(n_samples=30, n_steps=10, d=2)
        model = SBBTS(
            beta=50.0,
            n_steps=1,
            n_epochs=2,
            batch_size=16,
        )
        model.fit(X, verbose=False)
        return model, tmp_path / "model.pt"

    def test_save_load_roundtrip(self, fitted_model):
        """Model should be recoverable after save/load."""
        model, path = fitted_model
        model.save(path)

        loaded = SBBTS.load(path)
        assert loaded._fitted
        assert loaded.beta == model.beta
        assert loaded.d_model == model.d_model

    def test_loaded_model_can_sample(self, fitted_model):
        """Loaded model should be able to sample."""
        model, path = fitted_model
        model.save(path)

        loaded = SBBTS.load(path)
        X_synth = loaded.sample(n=5)
        assert X_synth.shape[0] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
