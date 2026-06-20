"""
Tests for new SBBTS features:
  - sample_conditional
  - covariates
  - compute_metrics / compute_tstr
  - PCAKMeansReducer
  - evaluate_augmentation (built-in TSTR)
  - seed reproducibility
  - lr_scheduler parameter
  - from_config()
  - PathSignatureEncoder warning
  - loss convergence (smoke)
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from sbbts.core.sbbts_solver import SBBTS


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_gbm(n_samples: int, n_steps: int, d: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_returns = 0.01 * rng.standard_normal((n_samples, n_steps, d))
    log_prices = np.concatenate(
        [np.zeros((n_samples, 1, d)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return log_prices.astype(np.float32)


def _fitted_model(n_samples=40, n_steps=10, d=2) -> SBBTS:
    X = _make_gbm(n_samples, n_steps, d)
    model = SBBTS(beta=50.0, n_steps=1, n_epochs=3, batch_size=16, seed=42, lr_scheduler="cosine")
    model.fit(X, verbose=False)
    return model, X


# ---------------------------------------------------------------------------
# 1. Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    def test_same_seed_same_samples(self):
        """Two models trained with same seed must produce identical samples."""
        X = _make_gbm(40, 10, 2)
        kwargs = dict(beta=50.0, n_steps=1, n_epochs=3, batch_size=16, seed=7)

        m1 = SBBTS(**kwargs)
        m1.fit(X, verbose=False)
        s1 = m1.sample(n=5)

        m2 = SBBTS(**kwargs)
        m2.fit(X, verbose=False)
        s2 = m2.sample(n=5)

        np.testing.assert_allclose(s1, s2, rtol=1e-4, atol=1e-5)

    def test_different_seeds_differ(self):
        """Models with different seeds should (almost certainly) differ."""
        X = _make_gbm(40, 10, 2)
        m1 = SBBTS(beta=50.0, n_steps=1, n_epochs=3, batch_size=16, seed=1)
        m1.fit(X, verbose=False)

        m2 = SBBTS(beta=50.0, n_steps=1, n_epochs=3, batch_size=16, seed=2)
        m2.fit(X, verbose=False)

        s1 = m1.sample(n=5)
        s2 = m2.sample(n=5)
        assert not np.allclose(s1, s2, atol=1e-3), "Different seeds produced identical samples"


# ---------------------------------------------------------------------------
# 2. LR scheduler
# ---------------------------------------------------------------------------

class TestLRScheduler:
    def test_cosine_scheduler_accepted(self):
        """lr_scheduler='cosine' should not raise."""
        model = SBBTS(beta=50.0, n_steps=1, n_epochs=3, batch_size=16, lr_scheduler="cosine")
        X = _make_gbm(30, 10, 2)
        model.fit(X, verbose=False)
        assert model._fitted

    def test_none_scheduler_accepted(self):
        """lr_scheduler='none' should not raise."""
        model = SBBTS(beta=50.0, n_steps=1, n_epochs=3, batch_size=16, lr_scheduler="none")
        X = _make_gbm(30, 10, 2)
        model.fit(X, verbose=False)
        assert model._fitted

    def test_invalid_scheduler_raises(self):
        with pytest.raises(ValueError, match="lr_scheduler"):
            SBBTS(lr_scheduler="step")

    def test_scheduler_saved_and_loaded(self, tmp_path):
        """lr_scheduler is persisted in checkpoint."""
        X = _make_gbm(30, 10, 2)
        model = SBBTS(beta=50.0, n_steps=1, n_epochs=2, batch_size=16, lr_scheduler="cosine")
        model.fit(X, verbose=False)
        p = tmp_path / "m.pt"
        model.save(p)
        loaded = SBBTS.load(p)
        assert loaded.lr_scheduler == "cosine"


# ---------------------------------------------------------------------------
# 3. Checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_checkpoint_files_created(self, tmp_path):
        """A checkpoint file per outer step should be written."""
        X = _make_gbm(40, 10, 2)
        model = SBBTS(beta=50.0, n_steps=2, n_epochs=2, batch_size=16)
        model.fit(X, verbose=False, checkpoint_dir=tmp_path)
        assert (tmp_path / "checkpoint_k1.pt").exists()
        assert (tmp_path / "checkpoint_k2.pt").exists()

    def test_checkpoint_loadable(self, tmp_path):
        """Checkpoints must be loadable and usable for sampling."""
        X = _make_gbm(40, 10, 2)
        model = SBBTS(beta=50.0, n_steps=2, n_epochs=2, batch_size=16)
        model.fit(X, verbose=False, checkpoint_dir=tmp_path)
        loaded = SBBTS.load(tmp_path / "checkpoint_k1.pt")
        assert loaded._fitted
        synth = loaded.sample(n=5)
        assert synth.shape[0] == 5

    def test_resume_from_step(self, tmp_path):
        """resume_from_step should skip already-trained outer steps."""
        X = _make_gbm(40, 10, 2)
        model = SBBTS(beta=50.0, n_steps=3, n_epochs=2, batch_size=16)
        model.fit(X, verbose=False, checkpoint_dir=tmp_path)

        # Resume from step 2 (only step 3 runs)
        loaded = SBBTS.load(tmp_path / "checkpoint_k2.pt")
        loaded.fit(X, verbose=False, resume_from_step=2)
        assert loaded._fitted


# ---------------------------------------------------------------------------
# 4. from_config()
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_default_config(self):
        """from_config() with no args should load defaults and return SBBTS."""
        model = SBBTS.from_config()
        assert isinstance(model, SBBTS)
        assert model.beta > 0

    def test_from_custom_yaml(self, tmp_path):
        """from_config() with a custom YAML should respect the values."""
        import yaml
        cfg = {"beta": 99.0, "n_steps": 3, "n_epochs": 5, "lr_scheduler": "none", "seed": 123}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        model = SBBTS.from_config(p)
        assert model.beta == 99.0
        assert model.n_steps == 3
        assert model.n_epochs == 5
        assert model.lr_scheduler == "none"
        assert model.seed == 123

    def test_from_config_can_fit(self, tmp_path):
        """Model built from config should be trainable."""
        import yaml
        cfg = {"beta": 50.0, "n_steps": 1, "n_epochs": 2, "batch_size": 16}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        model = SBBTS.from_config(p)
        X = _make_gbm(30, 10, 2)
        model.fit(X, verbose=False)
        assert model._fitted


# ---------------------------------------------------------------------------
# 5. sample_conditional
# ---------------------------------------------------------------------------

class TestSampleConditional:
    def test_output_shape(self):
        model, X = _fitted_model()
        n_total = X.shape[1]  # 11
        prefix = X[0, :4, :]  # (4, 2)
        result = model.sample_conditional(prefix, n=10)
        assert result.shape == (10, n_total, 2)

    def test_prefix_is_identical_across_samples(self):
        """All n continuations share the same observed prefix."""
        model, X = _fitted_model()
        prefix = X[0, :3, :]  # (3, 2)
        result = model.sample_conditional(prefix, n=5)
        # First 3 time steps must be identical across all 5 samples
        for i in range(1, 5):
            np.testing.assert_allclose(result[0, :3, :], result[i, :3, :], atol=1e-5)

    def test_continuation_varies(self):
        """The continuation part should vary across samples."""
        model, X = _fitted_model()
        prefix = X[0, :3, :]
        result = model.sample_conditional(prefix, n=5)
        # Steps 3 onwards should not all be equal
        assert not np.allclose(result[0, 3:, :], result[1, 3:, :], atol=1e-4)

    def test_prefix_too_long_raises(self):
        model, X = _fitted_model()
        with pytest.raises(ValueError, match="prefix length"):
            model.sample_conditional(X[0], n=3)  # full length = T, must be < T

    def test_batch_prefix_input(self):
        """Accepts (1, T_prefix, d) shaped prefix."""
        model, X = _fitted_model()
        prefix = X[0:1, :4, :]  # (1, 4, 2)
        result = model.sample_conditional(prefix, n=3)
        n_total = X.shape[1]
        assert result.shape == (3, n_total, 2)


# ---------------------------------------------------------------------------
# 6. Covariates
# ---------------------------------------------------------------------------

class TestCovariates:
    def test_fit_and_sample_with_covariates(self):
        """Model fitted with covariates should sample with same covariates."""
        n, T, d = 40, 10, 2
        cov_d = 3
        X = _make_gbm(n, T, d)
        rng = np.random.default_rng(0)
        covariates = rng.standard_normal((n, T + 1, cov_d)).astype(np.float32)

        model = SBBTS(
            beta=50.0, n_steps=1, n_epochs=3, batch_size=16, covariate_dim=cov_d
        )
        model.fit(X, covariates=covariates, verbose=False)

        n_synth = 8
        synth_cov = rng.standard_normal((n_synth, T + 1, cov_d)).astype(np.float32)
        result = model.sample(n=n_synth, covariates=synth_cov)
        assert result.shape == (n_synth, T + 1, d)

    def test_covariate_dim_mismatch_raises(self):
        """Sampling with wrong covariate dimension should fail at network level."""
        n, T, d, cov_d = 40, 10, 2, 3
        X = _make_gbm(n, T, d)
        rng = np.random.default_rng(0)
        covariates = rng.standard_normal((n, T + 1, cov_d)).astype(np.float32)

        model = SBBTS(
            beta=50.0, n_steps=1, n_epochs=2, batch_size=16, covariate_dim=cov_d
        )
        model.fit(X, covariates=covariates, verbose=False)

        wrong_cov = rng.standard_normal((5, T + 1, cov_d + 1)).astype(np.float32)
        with pytest.raises(Exception):
            model.sample(n=5, covariates=wrong_cov)


# ---------------------------------------------------------------------------
# 7. compute_metrics and compute_tstr
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    def test_compute_metrics_keys(self):
        from sbbts.utils.metrics import compute_metrics

        rng = np.random.default_rng(0)
        X_real = rng.standard_normal((50, 20, 2)).astype(np.float32)
        X_synth = rng.standard_normal((50, 20, 2)).astype(np.float32)
        result = compute_metrics(X_real, X_synth)

        for key in ("ann_std_real", "ann_std_synth", "ann_std_ratio",
                    "kurtosis_real", "kurtosis_synth", "acf_abs_sum_real"):
            assert key in result, f"Missing key: {key}"

    def test_compute_metrics_ratio_close_for_same_data(self):
        from sbbts.utils.metrics import compute_metrics

        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 20, 2)).astype(np.float32)
        # Using same data: all ratios should be ≈ 1
        result = compute_metrics(X, X.copy())
        assert abs(result["ann_std_ratio"] - 1.0) < 0.05

    def test_compute_tstr_keys(self):
        from sbbts.utils.metrics import compute_tstr

        rng = np.random.default_rng(2)
        X_real = rng.standard_normal((60, 15, 2)).astype(np.float32)
        X_synth = rng.standard_normal((60, 15, 2)).astype(np.float32)
        result = compute_tstr(X_real, X_synth)

        for key in ("trtr_mse", "tstr_mse", "ratio", "n_real_train", "n_real_test", "n_synth_train"):
            assert key in result, f"Missing key: {key}"

    def test_compute_tstr_ratio_near_1_for_same_dist(self):
        from sbbts.utils.metrics import compute_tstr

        rng = np.random.default_rng(3)
        # Real and synthetic drawn from same distribution → TSTR ≈ TRTR
        X_real = rng.standard_normal((200, 15, 1)).astype(np.float32)
        X_synth = rng.standard_normal((200, 15, 1)).astype(np.float32)
        result = compute_tstr(X_real, X_synth)
        assert result["ratio"] < 3.0, f"TSTR/TRTR ratio too large: {result['ratio']:.2f}"


# ---------------------------------------------------------------------------
# 8. PCAKMeansReducer
# ---------------------------------------------------------------------------

class TestPCAKMeansReducer:
    def test_fit_transform_shape(self):
        from sbbts.utils.dim_reduction import PCAKMeansReducer

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 20, 30)).astype(np.float32)
        reducer = PCAKMeansReducer(n_components=8, n_clusters=3)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (50, 20, 8)

    def test_inverse_transform_shape(self):
        from sbbts.utils.dim_reduction import PCAKMeansReducer

        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 20, 30)).astype(np.float32)
        reducer = PCAKMeansReducer(n_components=8, n_clusters=3, model_residuals=False)
        X_reduced = reducer.fit_transform(X)
        X_back = reducer.inverse_transform(X_reduced)
        assert X_back.shape == (50, 20, 30)

    def test_auto_n_components(self):
        """auto n_components selects via Marchenko-Pastur; n_clusters is clamped if needed."""
        from sbbts.utils.dim_reduction import PCAKMeansReducer

        rng = np.random.default_rng(2)
        # Large N, large T to give Marchenko-Pastur enough room for ≥ 1 component
        X = rng.standard_normal((200, 10, 20)).astype(np.float32)
        reducer = PCAKMeansReducer(n_components="auto", n_clusters=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape[0] == 200
        assert X_reduced.shape[2] >= 1

    def test_sbbts_with_dim_reducer(self):
        """SBBTS with PCAKMeansReducer: sample() must return original dimension."""
        from sbbts.utils.dim_reduction import PCAKMeansReducer

        rng = np.random.default_rng(3)
        # n_steps=10 → n_points=11; d_original=20, d_reduced=4
        X = rng.standard_normal((50, 10, 20)).astype(np.float32)
        reducer = PCAKMeansReducer(n_components=4, n_clusters=2, model_residuals=False)
        model = SBBTS(
            beta=50.0, n_steps=1, n_epochs=2, batch_size=16, dim_reducer=reducer
        )
        model.fit(X, verbose=False, check_input=False)
        synth = model.sample(n=5)
        # X has 10 time points → sample returns (n, 10, d_original=20)
        assert synth.shape == (5, 10, 20), f"Expected (5,10,20), got {synth.shape}"


# ---------------------------------------------------------------------------
# 9. evaluate_augmentation (built-in TSTR)
# ---------------------------------------------------------------------------

class TestEvaluateAugmentation:
    def test_default_tstr_keys(self):
        model, X = _fitted_model()
        n = len(X)
        X_train, X_test = X[: n // 2], X[n // 2:]
        result = model.evaluate_augmentation(X_train, X_test, augmentation_factor=2)
        assert "baseline" in result
        assert "augmented" in result
        assert "improvement" in result
        assert "tstr" in result

    def test_custom_downstream_fn(self):
        """backward-compat: custom downstream_model_fn still works."""
        model, X = _fitted_model()
        n = len(X)
        X_train, X_test = X[: n // 2], X[n // 2:]

        def dummy_fn(Xtr, Xte):
            return {"dummy_metric": float(Xtr.mean())}

        result = model.evaluate_augmentation(
            X_train, X_test, downstream_model_fn=dummy_fn, augmentation_factor=2
        )
        assert "dummy_metric" in result["baseline"]


# ---------------------------------------------------------------------------
# 10. PathSignatureEncoder warning
# ---------------------------------------------------------------------------

class TestSignatureEncoderWarning:
    def test_depth_reduction_emits_warning(self):
        """depth=2 auto-reduced to 1 should emit a UserWarning."""
        from sbbts.nn.signature_encoder import PathSignatureEncoder

        # d=40 → depth-2 sig dim = 40 + 40² = 1640 > default max_sig_dim=1024
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc = PathSignatureEncoder(input_dim=40, d_model=64, depth=2)
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("depth=2" in m and "depth=1" in m for m in msgs)
        assert enc.depth == 1

    def test_no_warning_when_depth_fits(self):
        from sbbts.nn.signature_encoder import PathSignatureEncoder

        # d=2 → depth-2 sig dim = 2 + 4 = 6 << max_sig_dim
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc = PathSignatureEncoder(input_dim=2, d_model=32, depth=2)
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) == 0
        assert enc.depth == 2


# ---------------------------------------------------------------------------
# 11. Loss convergence (smoke test)
# ---------------------------------------------------------------------------

class TestLossConvergence:
    def test_loss_decreases_over_epochs(self):
        """Training loss should decrease between first and last epoch."""
        import io, contextlib

        X = _make_gbm(60, 10, 2)
        losses = []

        class _LossCapture:
            def log(self, d):
                if "train_loss" in d and d.get("outer_step", 1) == 1:
                    losses.append(d["train_loss"])
            def section(self, *a): pass
            def write(self, *a): pass

        logger = _LossCapture()
        model = SBBTS(
            beta=50.0, n_steps=1, n_epochs=30, batch_size=16, seed=0, logger=logger
        )
        model.fit(X, verbose=False)

        assert len(losses) >= 10, "Not enough loss values captured"
        first_half_avg = np.mean(losses[: len(losses) // 2])
        second_half_avg = np.mean(losses[len(losses) // 2:])
        assert second_half_avg < first_half_avg, (
            f"Loss did not decrease: first_half={first_half_avg:.6f}, "
            f"second_half={second_half_avg:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
