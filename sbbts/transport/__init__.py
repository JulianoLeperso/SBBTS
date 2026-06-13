"""
Transport module for SBBTS.

Implements Brownian bridge sampling, transport maps, and conditional OT.
"""

from sbbts.transport.brownian_bridge import (
    sample_brownian_bridge,
    sample_brownian_bridge_batch,
    brownian_bridge_mean,
    brownian_bridge_std,
)
from sbbts.transport.transport_map import (
    x_to_y,
    y_to_x,
    validate_beta_condition,
    TransportMap,
)
from sbbts.transport.conditional_ot import (
    compute_conditional_transport,
    compute_score_target,
    compute_interval_loss,
    ConditionalOTSolver,
)

__all__ = [
    "sample_brownian_bridge",
    "sample_brownian_bridge_batch",
    "brownian_bridge_mean",
    "brownian_bridge_std",
    "x_to_y",
    "y_to_x",
    "validate_beta_condition",
    "TransportMap",
    "compute_conditional_transport",
    "compute_score_target",
    "compute_interval_loss",
    "ConditionalOTSolver",
]
