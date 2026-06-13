"""
Tests for Brownian Bridge implementation.

Validates:
1. E[y_t] = linear interpolation between y_start and y_end
2. Var[y_t] = σ²_t = (t - t_i)(t_{i+1} - t) / Δt_i
"""

import torch
import pytest
from sbbts.transport.brownian_bridge import (
    brownian_bridge_mean,
    brownian_bridge_std,
    sample_brownian_bridge,
    sample_brownian_bridge_batch,
)


class TestBrownianBridgeMean:
    """Test the mean computation of the Brownian bridge."""

    def test_endpoints(self):
        """Mean at endpoints should equal the endpoint values."""
        y_start = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_end = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        t_i, t_i1 = 0.0, 1.0

        mean_at_start = brownian_bridge_mean(y_start, y_end, t_i, t_i, t_i1)
        mean_at_end = brownian_bridge_mean(y_start, y_end, t_i1, t_i, t_i1)

        torch.testing.assert_close(mean_at_start, y_start)
        torch.testing.assert_close(mean_at_end, y_end)

    def test_midpoint(self):
        """Mean at midpoint should be average of endpoints."""
        y_start = torch.tensor([[0.0, 0.0]])
        y_end = torch.tensor([[2.0, 4.0]])
        t_i, t_i1 = 0.0, 1.0
        t_mid = 0.5

        mean_at_mid = brownian_bridge_mean(y_start, y_end, t_mid, t_i, t_i1)
        expected = (y_start + y_end) / 2

        torch.testing.assert_close(mean_at_mid, expected)

    def test_linear_interpolation(self):
        """Mean should be linear interpolation."""
        y_start = torch.tensor([[1.0]])
        y_end = torch.tensor([[3.0]])
        t_i, t_i1 = 0.0, 1.0

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mean = brownian_bridge_mean(y_start, y_end, t, t_i, t_i1)
            expected = y_start + t * (y_end - y_start)
            torch.testing.assert_close(mean, expected)


class TestBrownianBridgeStd:
    """Test the standard deviation computation of the Brownian bridge."""

    def test_endpoints_zero_variance(self):
        """Variance at endpoints should be zero."""
        t_i, t_i1 = 0.0, 1.0

        std_at_start = brownian_bridge_std(torch.tensor(t_i), t_i, t_i1)
        std_at_end = brownian_bridge_std(torch.tensor(t_i1), t_i, t_i1)

        assert std_at_start.item() == pytest.approx(0.0, abs=1e-7)
        assert std_at_end.item() == pytest.approx(0.0, abs=1e-7)

    def test_max_variance_at_midpoint(self):
        """Variance is maximized at the midpoint."""
        t_i, t_i1 = 0.0, 1.0
        dt = t_i1 - t_i

        t_mid = (t_i + t_i1) / 2
        std_mid = brownian_bridge_std(torch.tensor(t_mid), t_i, t_i1)
        expected_var_mid = (t_mid - t_i) * (t_i1 - t_mid) / dt
        expected_std_mid = expected_var_mid**0.5

        assert std_mid.item() == pytest.approx(expected_std_mid, rel=1e-6)

    def test_variance_formula(self):
        """Test σ²_t = (t - t_i)(t_{i+1} - t) / Δt_i."""
        t_i, t_i1 = 0.2, 0.8
        dt = t_i1 - t_i

        for t_val in [0.3, 0.5, 0.7]:
            t = torch.tensor(t_val)
            std = brownian_bridge_std(t, t_i, t_i1)
            expected_var = (t_val - t_i) * (t_i1 - t_val) / dt
            expected_std = expected_var**0.5

            assert std.item() == pytest.approx(expected_std, rel=1e-6)


class TestSampleBrownianBridge:
    """Test sampling from the Brownian bridge."""

    def test_mean_convergence(self):
        """Empirical mean should converge to theoretical mean."""
        torch.manual_seed(42)
        n_samples = 10000
        y_start = torch.tensor([[1.0, 2.0]]).expand(n_samples, -1)
        y_end = torch.tensor([[3.0, 5.0]]).expand(n_samples, -1)
        t_i, t_i1 = 0.0, 1.0
        t = 0.4

        samples = sample_brownian_bridge(y_start, y_end, t, t_i, t_i1)
        empirical_mean = samples.mean(dim=0)
        theoretical_mean = brownian_bridge_mean(y_start[0:1], y_end[0:1], t, t_i, t_i1).squeeze()

        torch.testing.assert_close(empirical_mean, theoretical_mean, atol=0.05, rtol=0.05)

    def test_variance_convergence(self):
        """Empirical variance should converge to theoretical variance."""
        torch.manual_seed(42)
        n_samples = 10000
        d = 3
        y_start = torch.zeros(n_samples, d)
        y_end = torch.zeros(n_samples, d)
        t_i, t_i1 = 0.0, 1.0
        t = 0.3

        samples = sample_brownian_bridge(y_start, y_end, t, t_i, t_i1)
        empirical_var = samples.var(dim=0)
        theoretical_var = (t - t_i) * (t_i1 - t) / (t_i1 - t_i)

        for var in empirical_var:
            assert var.item() == pytest.approx(theoretical_var, rel=0.05)

    def test_output_shape(self):
        """Test output shape matches input."""
        batch_size, d = 32, 5
        y_start = torch.randn(batch_size, d)
        y_end = torch.randn(batch_size, d)
        t = torch.rand(batch_size)
        t_i, t_i1 = 0.0, 1.0

        samples = sample_brownian_bridge(y_start, y_end, t, t_i, t_i1)

        assert samples.shape == (batch_size, d)


class TestSampleBrownianBridgeBatch:
    """Test batch sampling with uniform time."""

    def test_output_shapes(self):
        """Test output shapes for different n_samples."""
        batch_size, d = 16, 4
        y_start = torch.randn(batch_size, d)
        y_end = torch.randn(batch_size, d)
        t_i, t_i1 = 0.0, 1.0

        t, y_t = sample_brownian_bridge_batch(y_start, y_end, t_i, t_i1, n_samples=1)
        assert t.shape == (batch_size,)
        assert y_t.shape == (batch_size, d)

        t, y_t = sample_brownian_bridge_batch(y_start, y_end, t_i, t_i1, n_samples=5)
        assert t.shape == (batch_size, 5)
        assert y_t.shape == (batch_size, 5, d)

    def test_time_in_interval(self):
        """Sampled times should be in [t_i, t_{i+1})."""
        batch_size, d = 100, 2
        y_start = torch.randn(batch_size, d)
        y_end = torch.randn(batch_size, d)
        t_i, t_i1 = 0.2, 0.7

        t, _ = sample_brownian_bridge_batch(y_start, y_end, t_i, t_i1, n_samples=10)

        assert (t >= t_i).all()
        assert (t < t_i1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
