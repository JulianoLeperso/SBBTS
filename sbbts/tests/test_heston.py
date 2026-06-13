"""
Tests for Heston benchmark.

Validates:
1. Parameter sampling is within Table 3 ranges
2. Simulation produces valid trajectories
3. MLE estimation recovers reasonable parameters
"""

import numpy as np
import pytest
from sbbts.benchmarks.heston import (
    HestonParams,
    sample_heston_params,
    simulate_heston,
    simulate_heston_log_returns,
    generate_heston_dataset,
    estimate_heston_mle,
)


class TestHestonParams:
    """Test HestonParams dataclass."""

    def test_default_v0(self):
        """v0 should default to theta."""
        params = HestonParams(kappa=1.0, theta=0.5, xi=0.3, rho=-0.5, r=0.05)
        assert params.v0 == params.theta

    def test_custom_v0(self):
        """Should accept custom v0."""
        params = HestonParams(kappa=1.0, theta=0.5, xi=0.3, rho=-0.5, r=0.05, v0=0.3)
        assert params.v0 == 0.3

    def test_to_dict(self):
        """Should convert to dictionary."""
        params = HestonParams(kappa=1.0, theta=0.5, xi=0.3, rho=-0.5, r=0.05)
        d = params.to_dict()
        assert d["kappa"] == 1.0
        assert d["rho"] == -0.5


class TestSampleHestonParams:
    """Test parameter sampling."""

    def test_single_sample(self):
        """Should return single HestonParams."""
        params = sample_heston_params(n_samples=1, seed=42)
        assert isinstance(params, HestonParams)

    def test_multiple_samples(self):
        """Should return list for n_samples > 1."""
        params = sample_heston_params(n_samples=5, seed=42)
        assert isinstance(params, list)
        assert len(params) == 5

    def test_params_in_table3_ranges(self):
        """Parameters should be within Table 3 ranges."""
        params_list = sample_heston_params(n_samples=100, seed=42)
        for params in params_list:
            assert 0.5 <= params.kappa <= 4.0
            assert 0.5 <= params.theta <= 1.5
            assert 0.1 <= params.xi <= 0.9
            assert -0.9 <= params.rho <= 0.9
            assert 0.01 <= params.r <= 0.1


class TestSimulateHeston:
    """Test Heston simulation."""

    @pytest.fixture
    def sample_params(self):
        return HestonParams(
            kappa=2.0, theta=0.8, xi=0.5, rho=-0.7, r=0.05
        )

    def test_output_shape(self, sample_params):
        """Should return correct shape."""
        S, v = simulate_heston(sample_params, n_paths=10, n_steps=100, seed=42)
        assert S.shape == (10, 101)
        assert v.shape == (10, 101)

    def test_prices_positive(self, sample_params):
        """Prices should be positive."""
        S, v = simulate_heston(sample_params, n_paths=100, n_steps=252, seed=42)
        assert (S > 0).all()

    def test_variance_non_negative(self, sample_params):
        """Variance should be non-negative (truncated)."""
        S, v = simulate_heston(sample_params, n_paths=100, n_steps=252, seed=42)
        assert (v >= 0).all()

    def test_initial_values(self, sample_params):
        """Initial values should be S0 and v0."""
        S, v = simulate_heston(sample_params, n_paths=10, n_steps=50, S0=100.0, seed=42)
        assert np.allclose(S[:, 0], 100.0)
        assert np.allclose(v[:, 0], sample_params.v0)


class TestSimulateHestonLogReturns:
    """Test log return trajectory generation."""

    @pytest.fixture
    def sample_params(self):
        return HestonParams(
            kappa=2.0, theta=0.8, xi=0.5, rho=-0.7, r=0.05
        )

    def test_output_shape(self, sample_params):
        """Should return (n_paths, n_steps+1, 2)."""
        traj = simulate_heston_log_returns(sample_params, n_paths=10, n_steps=100, seed=42)
        assert traj.shape == (10, 101, 2)

    def test_initial_log_return_zero(self, sample_params):
        """Initial log return should be 0."""
        traj = simulate_heston_log_returns(sample_params, n_paths=10, n_steps=100, seed=42)
        assert np.allclose(traj[:, 0, 0], 0.0)


class TestGenerateHestonDataset:
    """Test dataset generation."""

    def test_heterogeneous_dataset(self):
        """Heterogeneous dataset should have different params per trajectory."""
        X, params = generate_heston_dataset(
            n_trajectories=10,
            trajectory_length=50,
            heterogeneous=True,
            seed=42,
        )
        assert X.shape == (10, 51, 2)
        assert len(params) == 10
        assert params[0].kappa != params[1].kappa

    def test_homogeneous_dataset(self):
        """Homogeneous dataset should have same params."""
        X, params = generate_heston_dataset(
            n_trajectories=10,
            trajectory_length=50,
            heterogeneous=False,
            seed=42,
        )
        assert X.shape == (10, 51, 2)
        assert params[0].kappa == params[1].kappa


class TestEstimateHestonMLE:
    """Test MLE estimation."""

    def test_estimates_within_reasonable_range(self):
        """Estimated parameters should be in reasonable ranges."""
        params = HestonParams(
            kappa=2.0, theta=0.8, xi=0.5, rho=-0.5, r=0.05
        )
        traj = simulate_heston_log_returns(params, n_paths=1, n_steps=1000, seed=42)[0]

        est = estimate_heston_mle(traj)

        assert 0 < est.kappa < 20
        assert 0 < est.theta < 5
        assert 0 < est.xi < 3
        assert -1 < est.rho < 1
        assert 0 <= est.r < 1

    def test_recovers_theta_approximately(self):
        """Should recover theta reasonably well."""
        params = HestonParams(
            kappa=2.0, theta=0.8, xi=0.3, rho=-0.3, r=0.05
        )
        traj = simulate_heston_log_returns(params, n_paths=1, n_steps=5000, seed=42)[0]

        est = estimate_heston_mle(traj)

        assert abs(est.theta - params.theta) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
