"""
Tests for transport map implementation.

Validates:
1. Y -> X is inverse of X -> Y at large β (approximately identity)
2. β * Δt > 1 condition is validated
3. Round-trip consistency: X -> Y -> X ≈ X
"""

import torch
import pytest
from sbbts.transport.transport_map import (
    x_to_y,
    y_to_x,
    validate_beta_condition,
    TransportMap,
)


class TestXToY:
    """Test forward transport map X -> Y."""

    def test_identity_at_infinite_beta(self):
        """For β → ∞, Y ≈ X (transport is identity)."""
        x = torch.randn(10, 3)
        score = torch.randn(10, 3)
        beta = 1e6

        y = x_to_y(x, score, beta)

        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_formula(self):
        """Test Y = X - (1/β) * score."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        score = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        beta = 10.0

        y = x_to_y(x, score, beta)
        expected = x - score / beta

        torch.testing.assert_close(y, expected)


class TestYToX:
    """Test inverse transport map Y -> X."""

    def test_formula(self):
        """Test X = Y + (1/β) * score."""
        y = torch.tensor([[1.0, 2.0]])
        score = torch.tensor([[0.5, 1.0]])
        beta = 5.0

        x = y_to_x(y, score, beta)
        expected = y + score / beta

        torch.testing.assert_close(x, expected)


class TestRoundTrip:
    """Test round-trip consistency of transport maps."""

    def test_round_trip_large_beta(self):
        """X -> Y -> X should approximately recover X for large β."""
        torch.manual_seed(42)
        x_orig = torch.randn(32, 5)
        beta = 100.0

        def mock_score_fn(t, x, ctx=None):
            return 0.1 * torch.sin(x)

        score_x = mock_score_fn(0.5, x_orig)
        y = x_to_y(x_orig, score_x, beta)

        score_y = mock_score_fn(0.5, y)
        x_recovered = y_to_x(y, score_y, beta)

        torch.testing.assert_close(x_recovered, x_orig, atol=1e-3, rtol=1e-3)

    def test_round_trip_moderate_beta(self):
        """Round-trip should still be close for moderate β."""
        torch.manual_seed(42)
        x_orig = torch.randn(16, 3)
        beta = 10.0

        score = torch.randn_like(x_orig) * 0.5
        y = x_to_y(x_orig, score, beta)
        x_recovered = y_to_x(y, score, beta)

        torch.testing.assert_close(x_recovered, x_orig, atol=1e-6, rtol=1e-6)


class TestValidateBetaCondition:
    """Test β * Δt > 1 validation."""

    def test_valid_condition(self):
        """Should not raise when β * Δt > 1."""
        validate_beta_condition(beta=10.0, dt=0.2)
        validate_beta_condition(beta=5.0, dt=0.5)
        validate_beta_condition(beta=100.0, dt=0.1)

    def test_invalid_condition_raises(self):
        """Should raise ValueError when β * Δt ≤ 1."""
        with pytest.raises(ValueError, match="Theorem 3.2 condition violated"):
            validate_beta_condition(beta=1.0, dt=0.5)

        with pytest.raises(ValueError, match="Theorem 3.2 condition violated"):
            validate_beta_condition(beta=2.0, dt=0.5)

    def test_error_message_includes_interval(self):
        """Error message should include interval index if provided."""
        with pytest.raises(ValueError, match="Interval 3"):
            validate_beta_condition(beta=1.0, dt=0.5, interval_idx=3)


class TestTransportMapClass:
    """Test the TransportMap class."""

    def test_init_validates_beta(self):
        """Should raise for non-positive β."""
        with pytest.raises(ValueError):
            TransportMap(beta=0.0)
        with pytest.raises(ValueError):
            TransportMap(beta=-1.0)

    def test_forward_with_precomputed_score(self):
        """Forward should work with pre-computed score."""
        tm = TransportMap(beta=10.0)
        x = torch.randn(8, 4)
        score = torch.randn_like(x)

        y = tm.forward(x, t=0.5, score=score)
        expected = x - score / 10.0

        torch.testing.assert_close(y, expected)

    def test_forward_requires_score_or_net(self):
        """Should raise if neither score nor score_net provided."""
        tm = TransportMap(beta=10.0)
        x = torch.randn(8, 4)

        with pytest.raises(ValueError, match="score or score_net"):
            tm.forward(x, t=0.5)

    def test_with_score_net(self):
        """Should work with score network."""
        def mock_score(t, x, context=None):
            return 0.1 * x

        tm = TransportMap(beta=10.0, score_net=mock_score)
        x = torch.randn(8, 4)

        y = tm.forward(x, t=0.5)
        expected = x - 0.1 * x / 10.0

        torch.testing.assert_close(y, expected)

    def test_inverse_with_score_net(self):
        """Inverse should work with score network."""
        def mock_score(t, y, context=None):
            return 0.2 * y

        tm = TransportMap(beta=5.0, score_net=mock_score)
        y = torch.randn(8, 4)

        x = tm.inverse(y, t=0.5)
        expected = y + 0.2 * y / 5.0

        torch.testing.assert_close(x, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
