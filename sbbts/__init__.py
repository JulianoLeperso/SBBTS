"""
SBBTS: Schrödinger–Bass Bridge for Time Series

A unified framework for synthetic financial time series generation that jointly
calibrates drift and volatility through optimal transport.

Based on the paper:
    "SBBTS: A Unified Schrödinger–Bass Framework for Synthetic Financial Time Series"
    Alouadi et al., 2026
"""

__version__ = "0.2.0"

from sbbts.core.sbbts_solver import SBBTS, init_distributed
from sbbts.utils.logger import SBBTSLogger
from sbbts.utils.visualization import (
    plot_acf_comparison,
    plot_marginal_comparison,
    plot_correlation_comparison,
    plot_sample_paths,
    plot_risk_metrics,
    plot_lag_corr_matrix,
    plot_qq,
    plot_acf_vol,
    plot_rolling_vol,
    plot_cluster_diagnostics,
    plot_leverage_effect,
    plot_signature_moments,
    tstr_score,
    diagnose,
    full_diagnose,
    # Generic (domain-agnostic) plots
    plot_feature_paths,
    plot_feature_marginals,
    plot_feature_acf,
    plot_feature_stats,
    diagnose_generic,
)
from sbbts.nn.inverse_net import InverseNet
from sbbts.nn.signature_encoder import PathSignatureEncoder
from sbbts.utils.early_stopping import EarlyStopping
from sbbts.utils.dim_reduction import (
    PCAKMeansReducer,
    PCAReducer,
    marchenko_pastur_n_components,
)
from sbbts.benchmarks.rough_volatility import (
    RoughHestonParams,
    simulate_rough_heston,
    generate_rough_heston_dataset,
)
from sbbts.utils.metrics import (
    log_ret_to_prices,
    compute_metrics,
    compute_tstr,
    compute_returns,
    var,
    expected_shortfall,
    sharpe_ratio,
    compute_all_risk_metrics,
    compute_generic_metrics,
)

__all__ = [
    # Model
    "SBBTS",
    "init_distributed",
    "SBBTSLogger",
    "__version__",
    # Neural network components
    "InverseNet",
    "PathSignatureEncoder",
    "EarlyStopping",
    # Dimensionality reduction
    "PCAKMeansReducer",
    "PCAReducer",
    "marchenko_pastur_n_components",
    # Benchmarks
    "RoughHestonParams",
    "simulate_rough_heston",
    "generate_rough_heston_dataset",
    # Visualization — individual panels
    "plot_acf_comparison",
    "plot_marginal_comparison",
    "plot_correlation_comparison",
    "plot_sample_paths",
    "plot_risk_metrics",
    "plot_lag_corr_matrix",
    "plot_qq",
    "plot_acf_vol",
    "plot_rolling_vol",
    "plot_cluster_diagnostics",
    "plot_leverage_effect",
    "plot_signature_moments",
    "tstr_score",
    # Visualization — composite figures
    "diagnose",
    "full_diagnose",
    # Visualization — generic (domain-agnostic)
    "plot_feature_paths",
    "plot_feature_marginals",
    "plot_feature_acf",
    "plot_feature_stats",
    "diagnose_generic",
    # Metrics & utilities
    "log_ret_to_prices",
    "compute_metrics",
    "compute_tstr",
    "compute_returns",
    "var",
    "expected_shortfall",
    "sharpe_ratio",
    "compute_all_risk_metrics",
    "compute_generic_metrics",
]
