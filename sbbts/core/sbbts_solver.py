"""
SBBTS Solver: Main algorithm for Schrödinger-Bass Bridge Time Series.

Implements Algorithm 1 from the paper with sklearn-like API:
    model = SBBTS(beta=10.0, n_steps=5)
    model.fit(X_train)
    X_synth = model.sample(n=500)

New features over baseline:
    - Low-β mode: InverseNet for stable training when β·Δt < low_beta_threshold
    - Early stopping with validation split
    - Signature encoder alternative to transformer
    - Conditional generation via external covariates
    - suggest_beta() static method for automatic β selection
    - diagnose() method for visual validation
    - Memory management during long sampling runs
    - Optional W&B/MLflow logging via logger= parameter
    - Cosine LR scheduler for better convergence
    - Automatic checkpointing between outer steps (checkpoint_dir)
    - Reproducibility via seed parameter
    - SBBTS.from_config(path) to build model from YAML
"""

import gc
import math
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm

from sbbts.core.score_network import ScoreNetwork, create_score_network
from sbbts.nn.inverse_net import InverseNet
from sbbts.nn.signature_encoder import PathSignatureEncoder
from sbbts.transport.transport_map import y_to_x
from sbbts.utils.early_stopping import EarlyStopping

# β·Δt threshold below which low-β InverseNet training is activated
_LOW_BETA_THRESHOLD = 3.0

# Sampling: free memory every N intervals to avoid OOM on long rollouts
_MEMORY_CLEANUP_INTERVAL = 50


def init_distributed(backend: str = "nccl") -> int:
    """
    Initialize torch.distributed for multi-GPU training.

    Call this once per process before model.fit().
    Reads RANK, LOCAL_RANK, WORLD_SIZE from the environment (set by torchrun).

    Launch with:
        torchrun --nproc_per_node=NUM_GPUS train_script.py

    Args:
        backend: 'nccl' for CUDA (recommended), 'gloo' for CPU

    Returns:
        local_rank (int) — which GPU this process owns
    """
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return local_rank


def load_default_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


class SBBTS(nn.Module):
    """
    Schrödinger-Bass Bridge for Time Series.

    Generative model for time series that jointly calibrates drift and
    volatility through optimal transport (Theorem 3.2 decomposition).

    Example:
        >>> model = SBBTS(beta=10.0, n_steps=5)
        >>> model.fit(X_train)                 # X_train: (N, T, d)
        >>> X_synth = model.sample(n=500)
        >>> X_aug   = model.augment(X_real, factor=200)
        >>> fig     = model.diagnose(X_real)
        >>> model.save("model.pt")

    Automatic β selection:
        >>> beta = SBBTS.suggest_beta(n_time_points=253, T=1.0)
        >>> model = SBBTS(beta=beta)
    """

    def __init__(
        self,
        beta: float = 10.0,
        n_steps: int = 5,
        d_model: int = 128,
        n_heads: int = 16,
        n_encoder_layers: int = 1,
        n_epochs: int = 1000,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        n_euler_steps: int = 50,
        t_tilde_offset: float = 0.01,
        device: str = None,
        dim_reducer=None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        compile_score_net: bool = False,
        use_fused_adam: bool = True,
        # Low-β mode
        low_beta_threshold: float = _LOW_BETA_THRESHOLD,
        n_inverse_epochs: int = 500,
        # Early stopping
        early_stopping_patience: int = 0,
        val_fraction: float = 0.1,
        # Encoder
        encoder_type: str = "transformer",
        signature_depth: int = 2,
        # Conditional generation
        covariate_dim: int = 0,
        # Optimisation stability
        grad_clip: float = 1.0,
        # LR scheduling
        lr_scheduler: str = "cosine",
        # Input normalisation
        normalize_input: bool = True,
        # Feature metadata
        feature_names: Optional[list] = None,
        # Reproducibility
        seed: Optional[int] = None,
        # Logging
        logger=None,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            beta: Regularization parameter β (must satisfy β·Δt > 1, Theorem 3.2)
            n_steps: Outer iterations K (Algorithm 1)
            d_model: Transformer latent dimension (Table 2: 128)
            n_heads: Attention heads (Table 2: 16)
            n_encoder_layers: Encoder layers (Appendix B: 1)
            n_epochs: Training epochs per K iteration (Table 2: 1000)
            batch_size: Mini-batch size (Table 2: 128)
            learning_rate: Adam learning rate (Table 2: 1e-3)
            n_euler_steps: Euler-Maruyama steps N_π (Table 2: 50)
            t_tilde_offset: Offset ξ for t̃_{i+1} = t_{i+1} - ξ·Δt
            device: 'cpu', 'cuda', or None (auto)
            dim_reducer: Optional PCAKMeansReducer for high-d data
            use_amp: Mixed precision (bfloat16) on CUDA
            amp_dtype: AMP dtype
            compile_score_net: torch.compile the score network
            use_fused_adam: Fused AdamW kernel on CUDA
            low_beta_threshold: β·Δt below this activates InverseNet training
            n_inverse_epochs: Epochs to train InverseNet per outer iteration
            early_stopping_patience: 0 = disabled; >0 = patience in epochs
            val_fraction: Fraction of data held out for early stopping validation
            encoder_type: 'transformer' (default) or 'signature' (path signatures)
            signature_depth: Truncation depth for signature encoder (1 or 2)
            covariate_dim: Dimension of external conditioning covariates (0 = none)
            grad_clip: Max gradient norm for clipping (0.0 = disabled, default 1.0).
                Prevents NaN loss from gradient explosion when scaling up d_model.
            normalize_input: If True (default), normalise X to zero-mean / unit-std
                per feature before training and undo the transform in sample().
                Strongly recommended for financial returns (scale ~0.01) to avoid
                the 100x scale mismatch with the N(0,1) sampling initialisation.
            feature_names: Optional list of d strings labelling each input feature.
                Stored in the model and propagated automatically to diagnose() and
                diagnose_generic(). Can be overridden at plot time. If None, names
                are auto-assigned in fit() as ["feature_0", ..., "feature_{d-1}"].
            lr_scheduler: Learning rate scheduling strategy applied within each outer
                step. Options: 'cosine' (CosineAnnealingLR, recommended) or 'none'
                (fixed lr). Cosine decay improves convergence in long training runs
                by gradually reducing lr from learning_rate to learning_rate/100.
            seed: Optional integer seed for full reproducibility. Sets torch and numpy
                random states at the start of fit(). Default None (non-deterministic).
            logger: Optional logger with .log(dict) method (W&B, MLflow, SBBTSLogger, etc.)
            log_dir: If set, auto-creates a SBBTSLogger in this directory and uses it
                as the run logger (ignored when logger= is explicitly provided).
        """
        super().__init__()

        if beta <= 0:
            raise ValueError(f"β must be positive, got {beta}")
        if encoder_type not in ("transformer", "signature"):
            raise ValueError(
                f"encoder_type must be 'transformer' or 'signature', got {encoder_type!r}"
            )
        if lr_scheduler not in ("cosine", "none"):
            raise ValueError(
                f"lr_scheduler must be 'cosine' or 'none', got {lr_scheduler!r}"
            )

        self.beta = beta
        self.n_steps = n_steps
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_euler_steps = n_euler_steps
        self.t_tilde_offset = t_tilde_offset
        self.dim_reducer = dim_reducer
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.compile_score_net = compile_score_net
        self.use_fused_adam = use_fused_adam
        self.low_beta_threshold = low_beta_threshold
        self.n_inverse_epochs = n_inverse_epochs
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.encoder_type = encoder_type
        self.signature_depth = signature_depth
        self.covariate_dim = covariate_dim
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.normalize_input = normalize_input
        self.seed = seed
        self.feature_names: Optional[list] = list(feature_names) if feature_names is not None else None
        # Filled by fit(); used to denormalise sample() output
        self._train_mean: Optional[Tensor] = None
        self._train_std: Optional[Tensor] = None

        if log_dir is not None and logger is None:
            from sbbts.utils.logger import SBBTSLogger

            logger = SBBTSLogger(base_dir=log_dir)
        self.logger = logger

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )

        self.score_net: Optional[ScoreNetwork] = None
        self.inverse_net: Optional[InverseNet] = None
        self.time_points: Optional[Tensor] = None
        self.input_dim: Optional[int] = None
        self._is_low_beta: bool = False
        self._fitted = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_beta(
        n_time_points: int,
        T: float = 1.0,
        safety_factor: float = 5.0,
    ) -> float:
        """
        Suggest a safe β value given data dimensions.

        Computes β such that β·Δt = safety_factor (default 5).
        Theorem 3.2 requires β·Δt > 1; safety_factor=5 gives comfortable margin.

        Args:
            n_time_points: Number of time steps (T dimension of data)
            T: Terminal time
            safety_factor: β·Δt target (1 = minimum valid, 5 = recommended)

        Returns:
            Suggested β value
        """
        dt = T / (n_time_points - 1)
        return safety_factor / dt

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = None) -> "SBBTS":
        """
        Instantiate SBBTS from a YAML configuration file.

        The YAML file maps to __init__ parameters. Unknown keys are ignored.
        When config_path is None, the built-in default.yaml is loaded.

        Example YAML::

            beta: 250.0
            n_steps: 5
            n_epochs: 1000
            batch_size: 128
            learning_rate: 0.001
            d_model: 128
            n_heads: 16
            n_encoder_layers: 1
            n_euler_steps: 50
            encoder_type: transformer
            lr_scheduler: cosine
            normalize_input: true
            seed: 42

        Args:
            config_path: Path to YAML file, or None to use default.yaml.

        Returns:
            Configured SBBTS instance (not yet fitted).

        Example:
            >>> model = SBBTS.from_config("my_config.yaml")
            >>> model.fit(X_train)
        """
        if config_path is None:
            cfg = load_default_config()
        else:
            with open(Path(config_path), "r") as f:
                cfg = yaml.safe_load(f) or {}

        # Flatten nested YAML sections (training:, network:, etc.) into a flat dict
        flat: dict = {}
        for v in cfg.values():
            if isinstance(v, dict):
                flat.update(v)
            # top-level scalars are also valid (flat YAML)
        for k, v in cfg.items():
            if not isinstance(v, dict):
                flat[k] = v

        # Map YAML keys → __init__ parameter names
        _key_map = {
            "K": "n_steps",
            "N_pi": "n_euler_steps",
            "activation": None,  # handled by architecture, not exposed
        }
        init_kwargs: dict = {}
        import inspect
        valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        for key, val in flat.items():
            mapped = _key_map.get(key, key)
            if mapped is not None and mapped in valid_params:
                init_kwargs[mapped] = val

        return cls(**init_kwargs)

    # ------------------------------------------------------------------
    # Data condition checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_data_conditions(X: "Tensor", feature_names: list) -> None:
        """
        Run pre-training sanity checks on the input data.

        Raises ValueError for hard failures; emits UserWarning for soft issues
        that may degrade quality but won't crash training.

        Checks (per feature):
          1. Near-constant  — std < 1e-6  → ValueError (model cannot learn)
          2. Non-stationary — ADF p > 0.05 → Warning (prices instead of returns?)
          3. Near-discrete  — n_unique / n_total < 0.01 → Warning (KL divergence risk)
          4. Extreme kurtosis — excess kurtosis > 100 → Warning (numerical instability risk)
          5. Scale mismatch — max_std / min_std > 1000 → Warning across features
        """
        import numpy as np_

        data = X.detach().cpu().numpy() if hasattr(X, "detach") else np_.asarray(X)
        N, T, d = data.shape
        flat = data.reshape(-1, d)  # (N*T, d) for marginal stats

        stds = flat.std(axis=0, ddof=1)

        # ── 1. Near-constant feature ────────────────────────────────────
        for i in range(d):
            if stds[i] < 1e-6:
                raise ValueError(
                    f"Feature '{feature_names[i]}' is near-constant (std={stds[i]:.2e}). "
                    f"SBBTS cannot learn a distribution from constant input. "
                    f"Remove or replace this feature."
                )

        # ── 2. Stationarity (ADF test, requires statsmodels) ────────────
        try:
            from statsmodels.tsa.stattools import adfuller
            _has_adf = True
        except ImportError:
            _has_adf = False

        if _has_adf:
            for i in range(d):
                # Use the aggregated per-window series: mean across windows at each time step
                series = data[:, :, i].mean(axis=0)  # (T,)
                if len(series) < 10:
                    continue
                try:
                    adf_stat, p_value, _, _, _, _ = adfuller(series, autolag="AIC", maxlag=min(10, T // 5))
                    if p_value > 0.05:
                        warnings.warn(
                            f"[SBBTS data check] Feature '{feature_names[i]}' may be non-stationary "
                            f"(ADF p={p_value:.3f} > 0.05). "
                            f"Non-stationary series violate the KL condition (Assumption 3.1). "
                            f"Consider differencing (e.g. log-returns instead of prices).",
                            UserWarning,
                            stacklevel=6,
                        )
                except Exception:
                    pass  # ADF can fail on degenerate series; silently skip
        else:
            # Fallback: simple linear trend test via OLS slope significance
            t_idx = np_.arange(T, dtype=np_.float64)
            t_centered = t_idx - t_idx.mean()
            for i in range(d):
                series = data[:, :, i].mean(axis=0)
                slope = np_.dot(t_centered, series) / (np_.dot(t_centered, t_centered) + 1e-12)
                trend_magnitude = abs(slope) * T / (stds[i] + 1e-12)
                if trend_magnitude > 2.0:
                    warnings.warn(
                        f"[SBBTS data check] Feature '{feature_names[i]}' shows a strong linear trend "
                        f"(trend/std ≈ {trend_magnitude:.1f}). "
                        f"Install statsmodels for a proper ADF stationarity test. "
                        f"Non-stationary input may violate the KL condition.",
                        UserWarning,
                        stacklevel=6,
                    )

        # ── 3. Near-discrete feature ────────────────────────────────────
        n_total = flat.shape[0]
        for i in range(d):
            n_unique = len(np_.unique(flat[:, i].round(6)))
            ratio = n_unique / n_total
            if ratio < 0.01:
                warnings.warn(
                    f"[SBBTS data check] Feature '{feature_names[i]}' appears nearly discrete "
                    f"({n_unique} unique values out of {n_total} observations, ratio={ratio:.4f}). "
                    f"SBBTS requires an absolutely continuous distribution (Assumption 3.1). "
                    f"Highly rounded or binary features may cause training instability.",
                    UserWarning,
                    stacklevel=6,
                )

        # ── 4. Extreme kurtosis ─────────────────────────────────────────
        try:
            from scipy.stats import kurtosis as _kurt
            for i in range(d):
                kurt = float(_kurt(flat[:, i], fisher=True))
                if kurt > 100:
                    warnings.warn(
                        f"[SBBTS data check] Feature '{feature_names[i]}' has extreme excess kurtosis "
                        f"({kurt:.1f} > 100). "
                        f"Very heavy tails may cause numerical instability in the score network. "
                        f"Consider winsorizing or log-transforming the feature.",
                        UserWarning,
                        stacklevel=6,
                    )
        except ImportError:
            pass

        # ── 5. Scale mismatch across features ───────────────────────────
        if d > 1:
            std_max = float(stds.max())
            std_min = float(stds[stds > 1e-12].min()) if (stds > 1e-12).any() else 1.0
            ratio = std_max / std_min
            if ratio > 1000:
                warnings.warn(
                    f"[SBBTS data check] Large scale mismatch across features "
                    f"(max_std / min_std = {ratio:.0f} > 1000). "
                    f"normalize_input=True (default) will rescale each feature independently, "
                    f"but verify that all features carry meaningful signal at their native scale.",
                    UserWarning,
                    stacklevel=6,
                )

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _validate_beta_for_data(self, n_time_points: int, T: float = 1.0) -> None:
        n_intervals = n_time_points - 1
        dt = T / n_intervals
        beta_dt = self.beta * dt

        if beta_dt <= 1.0:
            min_beta = 1.0 / dt + 0.1
            raise ValueError(
                f"Theorem 3.2 condition violated: β·Δt = {beta_dt:.4f} ≤ 1\n"
                f"For {n_intervals} intervals (Δt = {dt:.6f}), require β > {1/dt:.2f}\n"
                f"Suggested: set beta ≥ {self.suggest_beta(n_time_points, T):.1f} "
                f"(or call SBBTS.suggest_beta({n_time_points}, {T}))"
            )

        if beta_dt < self.low_beta_threshold:
            warnings.warn(
                f"β·Δt = {beta_dt:.3f} < {self.low_beta_threshold} (low-β threshold). "
                f"Activating InverseNet for stable training. "
                f"Consider β ≥ {self.suggest_beta(n_time_points, T, self.low_beta_threshold):.1f} "
                f"for high-β mode.",
                UserWarning,
                stacklevel=3,
            )
            self._is_low_beta = True
        else:
            self._is_low_beta = False

    def _create_time_points(self, n_time_points: int, T: float = 1.0) -> Tensor:
        return torch.linspace(0, T, n_time_points, device=self.device)

    def _init_score_network(self, input_dim: int) -> None:
        self.input_dim = input_dim
        encoder_input_dim = input_dim + self.covariate_dim

        if self.encoder_type == "signature":
            from sbbts.core.score_network import ScoreNetwork as SN
            from sbbts.nn.encoder import TrajectoryEncoder

            # Build ScoreNetwork but swap encoder
            self.score_net = create_score_network(
                input_dim=input_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_encoder_layers=self.n_encoder_layers,
                covariate_dim=self.covariate_dim,
            ).to(self.device)
            # Replace transformer encoder with signature encoder
            sig_enc = PathSignatureEncoder(
                input_dim=encoder_input_dim,
                d_model=self.d_model,
                depth=self.signature_depth,
            ).to(self.device)
            self.score_net.encoder = sig_enc
        else:
            self.score_net = create_score_network(
                input_dim=input_dim,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_encoder_layers=self.n_encoder_layers,
                covariate_dim=self.covariate_dim,
            ).to(self.device)

        if self.compile_score_net and hasattr(torch, "compile"):
            self.score_net = torch.compile(self.score_net, mode="reduce-overhead")

        if self._is_low_beta:
            self.inverse_net = InverseNet(
                input_dim=input_dim,
                d_model=max(32, self.d_model // 2),
            ).to(self.device)

    def _make_optimizer(self, params, fused: bool = True):
        adamw_kwargs = {"lr": self.learning_rate}
        if self.device.type == "cuda" and fused and self.use_fused_adam:
            adamw_kwargs["fused"] = True
        return torch.optim.AdamW(params, **adamw_kwargs)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_training_loss(
        self,
        batch: Tensor,
        time_points: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Score-matching loss for a batch (inner loop of Algorithm 1).

        Each adjacent pair (x[i], x[i+1]) defines a Brownian bridge parameterised
        in NORMALISED time s ∈ [0, T_bridge=1], matching the original paper repo.
        The denominator (T_bridge - s) is always ≥ t_tilde_offset (= safe_t = 0.01),
        keeping targets O(1–2) and gradient norms O(100) instead of O(1/dt_interval)
        which caused the 4500+ norms seen with actual-interval-time parameterisation.

        Args:
            batch: (B, N+1, d)
            time_points: (N+1,)  — used only to count n_intervals
            covariates: Optional (B, N+1, cov_d) conditioning series

        Returns:
            Scalar loss







        Changes Explications :
                1. The formula is unchanged — only the time parameterization of the bridge changed.

        The paper's Eq. (4.1) target is (Y_{t̃} − Y_t) / (t̃ − t). This is the score of a Brownian bridge from y_{t_i}
        to y_{t̃_{i+1}}. That is still exactly what we compute. What changed is the scale of the variable t:

        ┌───────────────────┬───────────────────────────────────┬───────────────────────────────┐
        │                   │            Old (ours)             │  New (ours = original repo)   │
        ├───────────────────┼───────────────────────────────────┼───────────────────────────────┤
        │ Bridge spans      │ [t_i, t_{i+1}], size 1/19 ≈ 0.053 │ [0, 1] — always, per interval │
        ├───────────────────┼───────────────────────────────────┼───────────────────────────────┤
        │ Denominator range │ [0.0005, 0.053]                   │ [0.01, 1.0]                   │
        ├───────────────────┼───────────────────────────────────┼───────────────────────────────┤
        │ Target scale      │ O(19)                             │ O(1–2)                        │
        ├───────────────────┼───────────────────────────────────┼───────────────────────────────┤
        │ Formula           │ (y_{t̃} − y_t) / (t̃ − t)           │ (y_T − y_s) / (T − s)         │
        └───────────────────┴───────────────────────────────────┴───────────────────────────────┘

        Same formula, different time axis. Parameterizing each interval's bridge in [0, 1] is a monotone re-scaling —
        it changes nothing mathematically but keeps the denominator bounded away from zero. The original repo does
        exactly this, which is why they never needed gradient clipping.

        Our old code was the literal reading of Eq. (4.1). The original repo's code is a numerically stable
        re-parameterization. They are equivalent in expectation.
        """
        batch_size, n_points, _ = batch.shape
        n_intervals = n_points - 1
        device, dtype = batch.device, batch.dtype

        x_ti = batch[:, :-1, :]  # (B, N, d)
        x_ti1 = batch[:, 1:, :]  # (B, N, d)

        # Encode all causal prefix contexts in one pass
        contexts = self.score_net.encode_all_prefixes(batch, covariates=covariates)

        # Normalised bridge duration — same for every interval, matching original repo.
        # safe_s keeps s strictly below T_bridge so the denominator never reaches 0.
        T_bridge = 1.0
        safe_s = self.t_tilde_offset  # default 0.01, same role as original safe_t

        # Transport X → Y at bridge-time s=0 (start) and s≈T_bridge (end).
        # The score network receives s ∈ [0, T_bridge], not actual calendar time.
        s_start = torch.zeros(batch_size, n_intervals, device=device, dtype=dtype)
        s_end = torch.full((batch_size, n_intervals), T_bridge - safe_s, device=device, dtype=dtype)

        score_at_start = self.score_net.forward_batched(s_start, x_ti, contexts)
        score_at_end = self.score_net.forward_batched(s_end, x_ti1, contexts)
        y_ti = x_ti - score_at_start / self.beta
        y_ti1 = x_ti1 - score_at_end / self.beta

        # Sample bridge time s ~ U[0, T_bridge - safe_s)
        s = torch.rand(batch_size, n_intervals, device=device, dtype=dtype) * (T_bridge - safe_s)

        # Brownian bridge from y_ti to y_ti1 at normalised time s:
        #   mean = (1 - s/T) y_ti + (s/T) y_ti1
        #   std  = sqrt(s (1 - s/T))          ← same formula as original get_loss
        alpha = (s / T_bridge).unsqueeze(-1)
        bridge_mean = (1.0 - alpha) * y_ti + alpha * y_ti1
        bridge_std = torch.sqrt((s * (1.0 - s / T_bridge)).clamp(min=0.0)).unsqueeze(-1)
        y_s = bridge_mean + bridge_std * torch.randn_like(bridge_mean)

        # Score-matching target: (y_ti1 - y_s) / (T_bridge - s)
        denom = (T_bridge - s).clamp(min=safe_s).unsqueeze(-1)
        target = (y_ti1 - y_s) / denom

        score_pred = self.score_net.forward_batched(s, y_s, contexts)
        return ((score_pred - target) ** 2).sum(dim=-1).mean()

    def _compute_inverse_loss(
        self,
        batch: Tensor,
        time_points: Tensor,
    ) -> Tensor:
        """
        Supervised loss for InverseNet in low-β mode.

        Target: InverseNet(t, Y_t) ≈ X_t - Y_t = (1/β) score(t, X_t, context)
        """
        batch_size, n_points, _ = batch.shape

        x_ti = batch[:, :-1, :]
        x_ti1 = batch[:, 1:, :]
        t_i = time_points[:-1].to(device=batch.device, dtype=batch.dtype)
        t_i1 = time_points[1:].to(device=batch.device, dtype=batch.dtype)

        with torch.no_grad():
            contexts = self.score_net.encode_all_prefixes(batch)
            score_ti = self.score_net.forward_batched(t_i, x_ti, contexts)
            score_ti1 = self.score_net.forward_batched(t_i1, x_ti1, contexts)
            y_ti = x_ti - score_ti / self.beta
            y_ti1 = x_ti1 - score_ti1 / self.beta
            target_ti = x_ti - y_ti  # = (1/β) score_ti
            target_ti1 = x_ti1 - y_ti1  # = (1/β) score_ti1

        pred_ti = self.inverse_net.forward_batched(t_i.unsqueeze(0).expand(batch_size, -1), y_ti)
        pred_ti1 = self.inverse_net.forward_batched(t_i1.unsqueeze(0).expand(batch_size, -1), y_ti1)

        return ((pred_ti - target_ti) ** 2).mean() + ((pred_ti1 - target_ti1) ** 2).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, Tensor],
        T: float = 1.0,
        verbose: bool = True,
        covariates: Optional[Union[np.ndarray, Tensor]] = None,
        resume_from_step: int = 0,
        feature_names: Optional[list] = None,
        check_input: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> "SBBTS":
        """
        Fit the SBBTS model (Algorithm 1).

        Args:
            X: Training trajectories, shape (N, T, d)
            T: Terminal time
            verbose: Show training progress
            covariates: Optional conditioning series, shape (N, T, cov_d).
                Used for conditional generation; model will only generate
                conditioned on the same covariate structure at sample time.
            resume_from_step: Start outer loop from this step index (0-based).
                Use after loading a checkpoint mid-training so already-learned
                weights are not reset.  Example: load a model saved after k=2,
                then call fit(X, resume_from_step=2) to continue from k=3.
            feature_names: Optional list of d strings labelling each input feature.
                Overrides the value passed to __init__ if provided here.
                Auto-generated as ["feature_0", ..., "feature_{d-1}"] if neither
                __init__ nor fit() supply names.
            check_input: If True (default), run pre-training sanity checks:
                stationarity (ADF), near-constant features, discreteness,
                extreme kurtosis, scale mismatch. Emits UserWarning or raises
                ValueError for hard failures. Set to False to skip all checks.
            checkpoint_dir: If set, the model is saved to this directory after
                each outer step k as ``checkpoint_k{k+1}.pt``. Allows resuming
                interrupted training via resume_from_step + SBBTS.load().

        Returns:
            self (fitted model)
        """
        # Reproducibility — seed pytorch and numpy before any random ops
        _seed = self.seed
        if _seed is not None:
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed)

        # Checkpoint directory setup
        _ckpt_dir: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if _ckpt_dir is not None:
            _ckpt_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        if covariates is not None:
            if isinstance(covariates, np.ndarray):
                covariates = torch.from_numpy(covariates).float()
            covariates = covariates.to(self.device)

        n_samples, n_time_points, input_dim = X.shape

        # Resolve feature names: fit() arg > __init__ arg > auto-generated
        if feature_names is not None:
            if len(feature_names) != input_dim:
                raise ValueError(
                    f"len(feature_names)={len(feature_names)} must equal d={input_dim}"
                )
            self.feature_names = list(feature_names)
        elif self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(input_dim)]

        if check_input:
            self._check_data_conditions(X, self.feature_names)

        self._validate_beta_for_data(n_time_points, T)
        self.time_points = self._create_time_points(n_time_points, T)

        if self.logger is not None and hasattr(self.logger, "section"):
            self.logger.section("SBBTS Model Configuration")
            self.logger.write(f"  beta={self.beta}  n_steps={self.n_steps}")
            self.logger.write(
                f"  d_model={self.d_model}  n_heads={self.n_heads}  n_encoder_layers={self.n_encoder_layers}"
            )
            self.logger.write(
                f"  n_epochs={self.n_epochs}  batch_size={self.batch_size}  lr={self.learning_rate}"
            )
            self.logger.write(f"  n_euler_steps={self.n_euler_steps}  device={self.device}")
            self.logger.write(
                f"  n_samples={n_samples}  n_time_points={n_time_points}  d={input_dim}"
            )
            self.logger.write(f"  low_beta_mode={self._is_low_beta}")
            self.logger.write(
                f"  normalize_input={self.normalize_input}  grad_clip={self.grad_clip}"
            )

        # Per-feature normalisation over the (N, T) axes.
        # Financial log-returns have std ~0.01 while the sampling initialisation
        # draws Y ~ N(0,1), creating a 100x scale mismatch that prevents the score
        # network from converging and causes gradient norms in the thousands.
        # Normalising to unit-std makes x and Y_0 live on the same scale.
        if self.normalize_input:
            self._train_mean = X.mean(dim=(0, 1), keepdim=True)  # (1, 1, d)
            self._train_std = X.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            X = (X - self._train_mean) / self._train_std
            if self.logger is not None and hasattr(self.logger, "write"):
                _m = self._train_mean.squeeze().tolist()
                _s = self._train_std.squeeze().tolist()
                self.logger.write(f"  train_mean={_m}  train_std={_s}")

        if self.dim_reducer is not None:
            X = (
                torch.from_numpy(self.dim_reducer.fit_transform(X.cpu().numpy()))
                .float()
                .to(self.device)
            )
            input_dim = X.shape[-1]

        self._init_score_network(input_dim)

        # ------------------------------------------------------------------
        # Distributed setup (pure CS optimization — gradient averaging is
        # mathematically equivalent to single-GPU with larger effective batch)
        # ------------------------------------------------------------------
        _dist = dist.is_initialized()
        _rank = dist.get_rank() if _dist else 0
        _world = dist.get_world_size() if _dist else 1
        _is_rank0 = _rank == 0

        if _dist:
            _ddp_kwargs = {}
            if self.device.type == "cuda":
                _ddp_kwargs["device_ids"] = [self.device.index or 0]
            self.score_net = nn.parallel.DistributedDataParallel(self.score_net, **_ddp_kwargs)
            if self._is_low_beta and self.inverse_net is not None:
                self.inverse_net = nn.parallel.DistributedDataParallel(
                    self.inverse_net, **_ddp_kwargs
                )

        # Validation split for early stopping (rank-0 only; others skip ES)
        use_es = self.early_stopping_patience > 0 and _is_rank0
        if use_es:
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val
            X_train, X_val = X[:n_train], X[n_val:]
            cov_train = covariates[:n_train] if covariates is not None else None
            cov_val = covariates[n_val:] if covariates is not None else None
        else:
            X_train, X_val = X, None
            cov_train, cov_val = covariates, None

        # Optimizer — points to DDP-wrapped params, which is correct
        optimizer = self._make_optimizer(self.score_net.parameters())
        inv_optimizer = (
            self._make_optimizer(self.inverse_net.parameters())
            if self._is_low_beta and self.inverse_net is not None
            else None
        )

        # LR scheduler — cosine annealing decays lr from learning_rate → lr/100
        # across n_epochs steps within each outer iteration, then resets.
        def _make_scheduler(opt):
            if self.lr_scheduler == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=self.n_epochs,
                    eta_min=self.learning_rate / 100,
                )
            return None

        scheduler = _make_scheduler(optimizer)
        inv_scheduler = _make_scheduler(inv_optimizer) if inv_optimizer is not None else None

        use_amp_flag = self.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp_flag)
        inv_scaler = torch.amp.GradScaler("cuda", enabled=use_amp_flag) if inv_optimizer else None

        train_dataset = (
            TensorDataset(X_train) if cov_train is None else TensorDataset(X_train, cov_train)
        )

        if _dist:
            # Each rank sees a disjoint shard; effective batch = batch_size × world_size
            sampler = DistributedSampler(
                train_dataset, num_replicas=_world, rank=_rank, shuffle=True, drop_last=True
            )
            dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            sampler = None
            dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
            )

        early_stopper = EarlyStopping(patience=self.early_stopping_patience) if use_es else None

        # Gate verbose and external logging to rank 0 only
        log_verbose = verbose and _is_rank0

        import time as _time

        for k in range(resume_from_step, self.n_steps):
            _step_t0 = _time.perf_counter()
            _step_losses: list = []
            _step_gradnorms: list = []

            if log_verbose:
                print(
                    f"\n=== Outer iteration k={k+1}/{self.n_steps}"
                    + (f" [{_world} GPUs]" if _dist else "")
                    + " ==="
                )
            if self.logger is not None and hasattr(self.logger, "section"):
                self.logger.section(f"Outer Iteration k={k+1}/{self.n_steps} — Training")

            if sampler is not None:
                sampler.set_epoch(k)  # ensures different shuffles across epochs in DDP

            epoch_iter = (
                tqdm(range(self.n_epochs), desc=f"K={k+1}") if log_verbose else range(self.n_epochs)
            )

            for epoch in epoch_iter:
                epoch_loss = 0.0
                epoch_max_gnorm = 0.0
                n_batches = 0

                for items in dataloader:
                    batch = items[0]
                    cov_batch = items[1] if len(items) > 1 else None
                    optimizer.zero_grad(set_to_none=True)

                    ctx = (
                        torch.amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp_flag
                        )
                        if use_amp_flag
                        else nullcontext()
                    )
                    with ctx:
                        loss = self._compute_training_loss(batch, self.time_points, cov_batch)

                    scaler.scale(loss).backward()
                    if self.grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        gnorm = float(
                            nn.utils.clip_grad_norm_(self.score_net.parameters(), self.grad_clip)
                        )
                        epoch_max_gnorm = max(epoch_max_gnorm, gnorm)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    n_batches += 1

                # Step LR scheduler once per epoch (after all mini-batches)
                if scheduler is not None:
                    scheduler.step()

                avg_loss = epoch_loss / max(n_batches, 1)
                _step_losses.append(avg_loss)
                if self.grad_clip > 0.0:
                    _step_gradnorms.append(epoch_max_gnorm)

                if log_verbose and (epoch + 1) % 100 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    tqdm.write(
                        f"  Epoch {epoch+1}: train_loss = {avg_loss:.6f}  "
                        f"grad_norm_max = {epoch_max_gnorm:.4f}  lr = {current_lr:.2e}"
                    )

                if _is_rank0 and self.logger is not None:
                    entry = {"outer_step": k + 1, "epoch": epoch + 1, "train_loss": avg_loss}
                    if self.grad_clip > 0.0:
                        entry["grad_norm"] = epoch_max_gnorm
                    self.logger.log(entry)

                # Early stopping — rank 0 computes, broadcasts stop signal to all ranks
                if use_es and early_stopper is not None:
                    _underlying = self.score_net.module if _dist else self.score_net
                    with torch.no_grad():
                        val_loss = self._compute_training_loss(
                            X_val, self.time_points, cov_val
                        ).item()
                    _stop = early_stopper(val_loss, _underlying)
                    if _dist:
                        _flag = torch.tensor(int(_stop), device=self.device)
                        dist.broadcast(_flag, src=0)
                        _stop = bool(_flag.item())
                    if _stop:
                        if log_verbose:
                            tqdm.write(
                                f"  Early stopping at epoch {epoch+1} (best val={early_stopper.best_loss:.6f})"
                            )
                        if _is_rank0:
                            early_stopper.load_best_weights(_underlying)
                        if _dist:
                            # Sync best weights from rank 0 to all other ranks
                            for param in _underlying.parameters():
                                dist.broadcast(param.data, src=0)
                        break

            # Low-β: train InverseNet after each outer iteration
            if self._is_low_beta and inv_optimizer is not None:
                if log_verbose:
                    tqdm.write(f"  [Low-β] Training InverseNet ({self.n_inverse_epochs} epochs)...")

                inv_iter = (
                    tqdm(range(self.n_inverse_epochs), desc="InvNet", leave=False)
                    if log_verbose
                    else range(self.n_inverse_epochs)
                )
                for _ in inv_iter:
                    if sampler is not None:
                        sampler.set_epoch(_)
                    for items in dataloader:
                        batch = items[0]
                        inv_optimizer.zero_grad(set_to_none=True)
                        ctx = (
                            torch.amp.autocast(
                                device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=use_amp_flag,
                            )
                            if use_amp_flag
                            else nullcontext()
                        )
                        with ctx:
                            inv_loss = self._compute_inverse_loss(batch, self.time_points)
                        inv_scaler.scale(inv_loss).backward()
                        inv_scaler.step(inv_optimizer)
                        inv_scaler.update()

            if early_stopper is not None:
                early_stopper.reset()

            # Reset LR scheduler at the start of each new outer step so cosine
            # decay restarts from learning_rate rather than continuing the descent.
            if scheduler is not None:
                scheduler = _make_scheduler(optimizer)
            if inv_scheduler is not None:
                inv_scheduler = _make_scheduler(inv_optimizer)

            # Automatic checkpointing — save model state after each outer step.
            # Use SBBTS.load(path) + fit(X, resume_from_step=k+1) to resume.
            if _ckpt_dir is not None and _is_rank0:
                _ckpt_path = _ckpt_dir / f"checkpoint_k{k+1}.pt"
                self._fitted = True  # mark fitted so save() works mid-training
                self.save(_ckpt_path)
                if log_verbose:
                    tqdm.write(f"  Checkpoint saved → {_ckpt_path}")

            # Write per-step convergence summary to diagnostics.log
            if (
                _is_rank0
                and self.logger is not None
                and hasattr(self.logger, "summarize_outer_step")
            ):
                _step_elapsed = _time.perf_counter() - _step_t0
                self.logger.summarize_outer_step(
                    k=k + 1,
                    n_steps=self.n_steps,
                    losses=_step_losses,
                    grad_norms=_step_gradnorms or None,
                    elapsed=_step_elapsed,
                    grad_clip=self.grad_clip,
                )

        # Unwrap DDP so sample() / save() / diagnose() work without modification
        if _dist:
            self.score_net = self.score_net.module
            if self._is_low_beta and self.inverse_net is not None:
                self.inverse_net = self.inverse_net.module

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        X_init: Union[np.ndarray, Tensor] = None,
        return_full_trajectory: bool = True,
        covariates: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> np.ndarray:
        """
        Generate synthetic time series.

        Args:
            n: Number of samples
            X_init: Optional initial states, shape (n, d) or (n, 1, d)
            return_full_trajectory: Return all T steps (True) or just final (False)
            covariates: Conditioning covariates, shape (n, T, cov_d). Required
                if model was fitted with covariate_dim > 0.

        Returns:
            Generated samples, shape (n, T, d) or (n, d)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before sampling")

        self.score_net.eval()
        if self.inverse_net is not None:
            self.inverse_net.eval()

        with torch.no_grad():
            if X_init is None:
                X_init = torch.randn(n, self.input_dim, device=self.device)
            else:
                if isinstance(X_init, np.ndarray):
                    X_init = torch.from_numpy(X_init).float()
                X_init = X_init.to(self.device)
                if X_init.dim() == 3:
                    X_init = X_init[:, 0, :]

            if covariates is not None:
                if isinstance(covariates, np.ndarray):
                    covariates = torch.from_numpy(covariates).float()
                covariates = covariates.to(self.device)

            n_points = len(self.time_points)
            n_intervals = n_points - 1
            trajectory = torch.zeros(n, n_points, self.input_dim, device=self.device)
            trajectory[:, 0, :] = X_init

            y_current = X_init.clone()
            cov_prefix = covariates[:, :1, :] if covariates is not None else None
            context = self.score_net.encode_trajectory(X_init.unsqueeze(1), covariates=cov_prefix)

            # Each interval uses normalised bridge time s ∈ [0, T_bridge=1],
            # matching the training loss.  dt_bridge is the same for every interval.
            _T_bridge = 1.0
            _safe_s = self.t_tilde_offset
            _dt_bridge = _T_bridge / self.n_euler_steps  # e.g. 1/50 = 0.02

            for i in range(n_intervals):
                # Euler-Maruyama in normalised bridge time s: 0 → T_bridge
                for step in range(self.n_euler_steps):
                    s_cur = step * _dt_bridge
                    s_tensor = torch.full((n,), s_cur, device=self.device)
                    drift = self.score_net.forward_with_context(s_tensor, y_current, context)
                    y_current = (
                        y_current
                        + drift * _dt_bridge
                        + torch.randn_like(y_current) * math.sqrt(_dt_bridge)
                    )
                    # Periodic memory cleanup inside the Euler loop to handle
                    # long trajectories with many Euler steps (n_euler_steps >> 50).
                    if (
                        self.device.type == "cuda"
                        and (step + 1) % _MEMORY_CLEANUP_INTERVAL == 0
                    ):
                        gc.collect()
                        torch.cuda.empty_cache()

                # Recover X from Y at bridge-time s ≈ T_bridge
                s_end_tensor = torch.full((n,), _T_bridge - _safe_s, device=self.device)
                if self._is_low_beta and self.inverse_net is not None:
                    correction = self.inverse_net(s_end_tensor, y_current)
                    x_ti1 = y_current + correction
                else:
                    score_at_end = self.score_net.forward_with_context(
                        s_end_tensor, y_current, context
                    )
                    x_ti1 = y_to_x(y_current, score_at_end, self.beta)

                trajectory[:, i + 1, :] = x_ti1

                # Update context for next interval
                if i < n_intervals - 1:
                    cov_so_far = covariates[:, : i + 2, :] if covariates is not None else None
                    context = self.score_net.encode_trajectory(
                        trajectory[:, : i + 2, :], covariates=cov_so_far
                    )

                # Cleanup after each interval (handles long trajectory rollouts)
                if (i + 1) % _MEMORY_CLEANUP_INTERVAL == 0 and self.device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

        self.score_net.train()
        if self.inverse_net is not None:
            self.inverse_net.train()

        result = trajectory.cpu().numpy()

        # Inverse operations must be applied in reverse training order:
        # training: X_orig → normalize → X_norm → dim_reduce → X_reduced (→ score net)
        # sampling: X_reduced → inv_dim_reduce → X_norm → denormalize → X_orig
        if self.dim_reducer is not None:
            result = self.dim_reducer.inverse_transform(result)

        if self.normalize_input and self._train_std is not None:
            result = result * self._train_std.cpu().numpy() + self._train_mean.cpu().numpy()

        return result if return_full_trajectory else result[:, -1, :]

    def sample_conditional(
        self,
        X_prefix: Union[np.ndarray, Tensor],
        n: int = 1,
    ) -> np.ndarray:
        """
        Generate continuations of a given prefix trajectory.

        Given an observed prefix of length T_prefix < T, generates n plausible
        completions of the remaining T - T_prefix steps.  Useful for fan charts,
        conditional stress tests, and backtesting.

        Args:
            X_prefix: Observed prefix, shape (T_prefix, d) or (1, T_prefix, d).
                      Must be in the same scale as the data passed to fit().
            n:        Number of continuations to generate.

        Returns:
            Full trajectories of shape (n, T, d) — prefix is identical in all
            n samples; only the continuation differs.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling sample_conditional")

        if isinstance(X_prefix, np.ndarray):
            X_prefix = torch.from_numpy(X_prefix).float()
        X_prefix = X_prefix.to(self.device)

        if X_prefix.dim() == 2:
            X_prefix = X_prefix.unsqueeze(0)  # (1, T_prefix, d)

        T_prefix = X_prefix.shape[1]
        n_points = len(self.time_points)
        if T_prefix >= n_points:
            raise ValueError(f"prefix length {T_prefix} must be shorter than T={n_points}")

        self.score_net.eval()
        if self.inverse_net is not None:
            self.inverse_net.eval()

        with torch.no_grad():
            # Normalise prefix the same way fit() normalised training data.
            if self.normalize_input and self._train_std is not None:
                X_prefix = (X_prefix - self._train_mean) / self._train_std

            # Repeat prefix n times → (n, T_prefix, d)
            prefix_batch = X_prefix.expand(n, -1, -1).clone()

            n_intervals = n_points - 1
            _T_bridge = 1.0
            _safe_s = self.t_tilde_offset
            _dt_bridge = _T_bridge / self.n_euler_steps

            trajectory = torch.zeros(n, n_points, self.input_dim, device=self.device)
            trajectory[:, :T_prefix, :] = prefix_batch

            # Encode full prefix as context for the first new interval.
            context = self.score_net.encode_trajectory(prefix_batch)

            # Compute Y at the last prefix step using the actual transport map:
            #   Y = X - (1/β) · s_θ(t≈T_bridge, X, context)
            # This is exact up to the large-β approximation used everywhere else,
            # but is far more accurate than the naive Y ≈ X when β is moderate.
            _last_x = prefix_batch[:, -1, :]
            _s_end_for_prefix = torch.full((n,), _T_bridge - _safe_s, device=self.device)
            with torch.no_grad():
                _score_last = self.score_net.forward_with_context(
                    _s_end_for_prefix, _last_x, context
                )
            if self._is_low_beta and self.inverse_net is not None:
                # In low-β mode InverseNet gives X→Y direction as well
                y_current = _last_x - self.inverse_net(_s_end_for_prefix, _last_x)
            else:
                from sbbts.transport.transport_map import x_to_y
                y_current = x_to_y(_last_x, _score_last, self.beta)

            for i in range(T_prefix - 1, n_intervals):
                for step in range(self.n_euler_steps):
                    s_cur = step * _dt_bridge
                    s_tensor = torch.full((n,), s_cur, device=self.device)
                    drift = self.score_net.forward_with_context(s_tensor, y_current, context)
                    y_current = (
                        y_current
                        + drift * _dt_bridge
                        + torch.randn_like(y_current) * math.sqrt(_dt_bridge)
                    )

                s_end_tensor = torch.full((n,), _T_bridge - _safe_s, device=self.device)
                if self._is_low_beta and self.inverse_net is not None:
                    x_ti1 = y_current + self.inverse_net(s_end_tensor, y_current)
                else:
                    from sbbts.transport.transport_map import y_to_x

                    score_at_end = self.score_net.forward_with_context(
                        s_end_tensor, y_current, context
                    )
                    x_ti1 = y_to_x(y_current, score_at_end, self.beta)

                trajectory[:, i + 1, :] = x_ti1

                if i < n_intervals - 1:
                    context = self.score_net.encode_trajectory(trajectory[:, : i + 2, :])

                if (i + 1) % _MEMORY_CLEANUP_INTERVAL == 0 and self.device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

        self.score_net.train()
        if self.inverse_net is not None:
            self.inverse_net.train()

        result = trajectory.cpu().numpy()

        # Same inverse order as sample(): dim_reduce first, then denormalize
        if self.dim_reducer is not None:
            result = self.dim_reducer.inverse_transform(result)

        if self.normalize_input and self._train_std is not None:
            result = result * self._train_std.cpu().numpy() + self._train_mean.cpu().numpy()

        return result

    def sample_batches(
        self,
        n: int,
        batch_size: int = 500,
    ):
        """
        Generate n synthetic trajectories in batches to avoid OOM.

        A generator that yields np.ndarray batches of shape (B, T, d) where
        B ≤ batch_size.  Useful when n is large (e.g. 100 000+) and the full
        tensor would not fit in RAM or GPU memory.

        Args:
            n:          Total number of trajectories to generate.
            batch_size: Maximum trajectories per yielded batch.

        Yields:
            np.ndarray of shape (B, T, d)

        Example::

            chunks = []
            for batch in model.sample_batches(n=50_000, batch_size=500):
                chunks.append(batch)
            X_synth = np.concatenate(chunks, axis=0)   # (50000, T, d)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling sample_batches")
        generated = 0
        while generated < n:
            current = min(batch_size, n - generated)
            yield self.sample(n=current)
            generated += current

    # ------------------------------------------------------------------
    # Augmentation & evaluation
    # ------------------------------------------------------------------

    def augment(
        self,
        X_real: Union[np.ndarray, Tensor],
        factor: int = 200,
    ) -> np.ndarray:
        """
        Augment real data with factor×N synthetic samples.

        Args:
            X_real: Real trajectories, shape (N, T, d)
            factor: Augmentation factor (Table 1: 200)

        Returns:
            Augmented dataset, shape (N + factor*N, T, d)
        """
        if isinstance(X_real, Tensor):
            X_real = X_real.cpu().numpy()

        n_synth = factor * len(X_real)
        X_synth = self.sample(n=n_synth)
        return np.concatenate([X_real, X_synth], axis=0)

    def diagnose(
        self,
        X_real: Union[np.ndarray, Tensor],
        n_synth: int = 200,
        figsize: tuple = (16, 14),
        title: str = "SBBTS Diagnostic Report",
    ):
        """
        Generate a comprehensive diagnostic figure (real vs. synthetic).

        Plots sample paths, return distributions, ACF, correlation matrices,
        and risk metrics side-by-side.

        Requires matplotlib: pip install 'sbbts[viz]'

        Args:
            X_real: Real data, shape (N, T, d)
            n_synth: Number of synthetic samples to generate
            figsize: Figure size
            title: Figure title

        Returns:
            matplotlib.Figure
        """
        from sbbts.utils.visualization import diagnose as _diagnose

        if isinstance(X_real, Tensor):
            X_real = X_real.cpu().numpy()

        X_synth = self.sample(n=n_synth)
        return _diagnose(X_real, X_synth, figsize=figsize, title=title)

    def diagnose_generic(
        self,
        X_real: Union[np.ndarray, Tensor],
        n_synth: int = 200,
        feature_names: Optional[list] = None,
        figsize: tuple = None,
        title: str = "SBBTS Generic Diagnostic",
        max_lag: int = 20,
        n_paths: int = 5,
        max_cols: int = 3,
    ):
        """
        Domain-agnostic diagnostic figure (no financial assumptions).

        Generates a composite figure with per-feature panels for sample paths,
        marginal distributions, ACF, and cross-feature correlation — suitable
        for any multivariate time series (returns, volatility, macro factors, etc.).

        Args:
            X_real:        Real data, shape (N, T, d).
            n_synth:       Number of synthetic samples to generate.
            feature_names: Feature labels. Falls back to self.feature_names if None.
            figsize:       Figure size. Auto-computed from d if None.
            title:         Figure title.
            max_lag:       ACF lag count.
            n_paths:       Sample trajectories per panel.
            max_cols:      Grid columns per row of feature panels.

        Returns:
            matplotlib.Figure
        """
        from sbbts.utils.visualization import diagnose_generic as _dg

        if isinstance(X_real, Tensor):
            X_real = X_real.cpu().numpy()

        names = feature_names if feature_names is not None else self.feature_names
        X_synth = self.sample(n=n_synth)
        return _dg(
            X_real, X_synth,
            feature_names=names,
            figsize=figsize,
            title=title,
            max_lag=max_lag,
            n_paths=n_paths,
            max_cols=max_cols,
        )

    def evaluate_augmentation(
        self,
        X_train: Union[np.ndarray, Tensor],
        X_test: Union[np.ndarray, Tensor],
        downstream_model_fn=None,
        augmentation_factor: int = 5,
        ar_order: int = 5,
    ) -> dict:
        """
        Evaluate whether data augmentation improves a downstream model.

        Trains a model with and without augmentation and compares performance
        on X_test.  By default uses TSTR (Train-on-Synthetic, Test-on-Real)
        with a linear AR(ar_order) ridge regression — no downstream_model_fn
        needed.  A custom function can still be provided for domain-specific
        evaluations.

        Args:
            X_train: Training data, shape (N, T, d)
            X_test: Test data, shape (M, T, d)
            downstream_model_fn: Optional callable (X_train, X_test) -> metrics_dict.
                When None (default), uses the built-in TSTR metric
                (compute_tstr) which requires scikit-learn.
            augmentation_factor: Factor of synthetic samples added (e.g. 5 →
                5 × N synthetic samples mixed with N real samples).
            ar_order: AR lag order used by the built-in TSTR evaluator.

        Returns:
            dict with keys:

            * ``baseline``   — metrics on real training data only
            * ``augmented``  — metrics on augmented (real + synthetic) data
            * ``improvement`` — per-metric delta (augmented − baseline)
            * ``tstr``        — full compute_tstr dict (only when default evaluator)
        """
        from sbbts.utils.metrics import compute_tstr

        if isinstance(X_train, Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(X_test, Tensor):
            X_test = X_test.cpu().numpy()

        X_aug = self.augment(X_train, factor=augmentation_factor)

        if downstream_model_fn is None:
            # Built-in TSTR: AR(p) ridge trained on synth-only, evaluated on real test.
            # We also compute TRTR (train-on-real, test-on-real) as a baseline.
            n_synth = augmentation_factor * len(X_train)
            X_synth = X_aug[len(X_train):]  # augment prepends real, synth is the tail

            tstr_result = compute_tstr(
                X_real=np.concatenate([X_train, X_test], axis=0),
                X_synth=X_synth,
                ar_order=ar_order,
                test_fraction=len(X_test) / max(len(X_train) + len(X_test), 1),
            )
            baseline_metrics = {"ar_mse": tstr_result["trtr_mse"]}
            augmented_metrics = {"ar_mse": tstr_result["tstr_mse"]}
            improvement = {"ar_mse": tstr_result["tstr_mse"] - tstr_result["trtr_mse"]}
            return {
                "baseline": baseline_metrics,
                "augmented": augmented_metrics,
                "improvement": improvement,
                "tstr": tstr_result,
            }

        # Custom downstream evaluator path (backward-compatible)
        baseline_metrics = downstream_model_fn(X_train, X_test)
        augmented_metrics = downstream_model_fn(X_aug, X_test)

        improvement = {}
        for key in baseline_metrics:
            if key in augmented_metrics:
                try:
                    improvement[key] = float(augmented_metrics[key]) - float(baseline_metrics[key])
                except (TypeError, ValueError):
                    pass

        return {
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "improvement": improvement,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model to file."""
        path = Path(path)
        checkpoint = {
            "state_dict": self.score_net.state_dict() if self.score_net else None,
            "inverse_state_dict": self.inverse_net.state_dict() if self.inverse_net else None,
            "config": {
                "beta": self.beta,
                "n_steps": self.n_steps,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_encoder_layers": self.n_encoder_layers,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "n_euler_steps": self.n_euler_steps,
                "t_tilde_offset": self.t_tilde_offset,
                "input_dim": self.input_dim,
                "use_amp": self.use_amp,
                "compile_score_net": self.compile_score_net,
                "use_fused_adam": self.use_fused_adam,
                "low_beta_threshold": self.low_beta_threshold,
                "n_inverse_epochs": self.n_inverse_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "val_fraction": self.val_fraction,
                "encoder_type": self.encoder_type,
                "signature_depth": self.signature_depth,
                "covariate_dim": self.covariate_dim,
                "is_low_beta": self._is_low_beta,
                "normalize_input": self.normalize_input,
                "lr_scheduler": self.lr_scheduler,
                "seed": self.seed,
            },
            "time_points": self.time_points,
            "fitted": self._fitted,
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "feature_names": self.feature_names,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = None) -> "SBBTS":
        """Load model from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]

        model = cls(
            beta=cfg["beta"],
            n_steps=cfg["n_steps"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_encoder_layers=cfg["n_encoder_layers"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            n_euler_steps=cfg["n_euler_steps"],
            t_tilde_offset=cfg["t_tilde_offset"],
            device=device,
            use_amp=cfg.get("use_amp", True),
            compile_score_net=cfg.get("compile_score_net", False),
            use_fused_adam=cfg.get("use_fused_adam", True),
            low_beta_threshold=cfg.get("low_beta_threshold", _LOW_BETA_THRESHOLD),
            n_inverse_epochs=cfg.get("n_inverse_epochs", 500),
            early_stopping_patience=cfg.get("early_stopping_patience", 0),
            val_fraction=cfg.get("val_fraction", 0.1),
            encoder_type=cfg.get("encoder_type", "transformer"),
            signature_depth=cfg.get("signature_depth", 2),
            covariate_dim=cfg.get("covariate_dim", 0),
            normalize_input=cfg.get("normalize_input", False),
            lr_scheduler=cfg.get("lr_scheduler", "cosine"),
            seed=cfg.get("seed", None),
        )

        model._is_low_beta = cfg.get("is_low_beta", False)
        model._train_mean = checkpoint.get("train_mean")
        model._train_std = checkpoint.get("train_std")

        if cfg["input_dim"] is not None:
            model._init_score_network(cfg["input_dim"])
            if checkpoint["state_dict"] is not None:
                model.score_net.load_state_dict(checkpoint["state_dict"])
            if model.inverse_net is not None and checkpoint.get("inverse_state_dict"):
                model.inverse_net.load_state_dict(checkpoint["inverse_state_dict"])

        model.time_points = checkpoint["time_points"]
        model._fitted = checkpoint["fitted"]
        model.feature_names = checkpoint.get("feature_names")
        return model
