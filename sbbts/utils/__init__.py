"""
Utility module for SBBTS.

Contains metrics, sampling utilities, dimensionality reduction, and visualization.
"""

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
)
