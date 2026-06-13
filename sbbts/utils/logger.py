"""
Two-file run logger for SBBTS.

Every run produces two files inside technical_logs/<timestamp>/:

  training_epochs.log  — one line per epoch; machine-readable; safe to grep/tail
  diagnostics.log      — human-readable structured report: model config, per-step
                         convergence summaries, gradient health, and all
                         post-training diagnostic tables written by viz functions

Usage
-----
    from sbbts.utils.logger import SBBTSLogger

    # Auto-created by SBBTS when log_dir is set:
    model = SBBTS(..., log_dir="technical_logs")

    # Or create manually and share:
    logger = SBBTSLogger(base_dir="technical_logs")
    model = SBBTS(..., logger=logger)

    # Pass to visualization functions for post-training diagnostics:
    plot_cluster_diagnostics(X_train, X_synth, logger=logger)
    plot_rolling_vol(log_rets, synth_rets, logger=logger)
    plot_signature_moments(X_train, X_synth, logger=logger)
"""

import datetime
import statistics
import sys
from pathlib import Path
from typing import List, Optional


class SBBTSLogger:
    """
    Two-file diagnostic logger.

    training_epochs.log
        One [TRAIN] line per epoch — never grows unbounded per diagnostic run,
        easy to grep for "nan" or tail for live monitoring.

    diagnostics.log
        The real debug tool. Structured sections written by the training loop
        (model config, per-outer-step convergence summary with gradient health)
        and by every visualization function (cluster coverage, vol ratio,
        signature moment ratios). Open this file after a run to understand
        what went right or wrong without reading thousands of epoch lines.
    """

    def __init__(
        self,
        base_dir: str = "technical_logs",
        run_name: Optional[str] = None,
    ) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = run_name if run_name else f"run_{ts}"
        self.run_dir = Path(base_dir) / name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.epochs_path = self.run_dir / "training_epochs.log"
        self.diag_path   = self.run_dir / "diagnostics.log"
        self.log_path    = self.diag_path          # backward-compat alias

        self._ef = open(self.epochs_path, "w", encoding="utf-8", buffering=1)
        self._df = open(self.diag_path,   "w", encoding="utf-8", buffering=1)

        self._write_headers(ts)

    # ── headers ──────────────────────────────────────────────────────────────

    def _write_headers(self, ts: str) -> None:
        self._ef.write("=" * 70 + "\n")
        self._ef.write(f"SBBTS training_epochs.log — {ts}\n")
        self._ef.write(f"Python {sys.version.split()[0]}\n")
        self._ef.write("Columns: [TRAIN] outer_step=K | epoch=E | train_loss=L | grad_norm=G\n")
        self._ef.write("=" * 70 + "\n\n")

        self._df.write("=" * 70 + "\n")
        self._df.write(f"SBBTS diagnostics.log — {ts}\n")
        self._df.write(f"Python {sys.version.split()[0]}\n")
        self._df.write("=" * 70 + "\n\n")

    # ── W&B / MLflow-compatible interface → training_epochs.log ──────────────

    def log(self, data: dict) -> None:
        """Called by SBBTS training loop each epoch. Writes to training_epochs.log."""
        parts = [
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in data.items()
        ]
        self._ef.write(f"[TRAIN] {' | '.join(parts)}\n")

    # ── diagnostics.log writers ───────────────────────────────────────────────

    def section(self, title: str) -> None:
        self._df.write(f"\n{'─' * 60}\n{title}\n{'─' * 60}\n")

    def write(self, text: str) -> None:
        self._df.write(text + "\n")

    def write_table(self, headers: list, rows: list) -> None:
        col_w = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        self._df.write(fmt.format(*[str(h) for h in headers]) + "\n")
        self._df.write("  ".join("-" * w for w in col_w) + "\n")
        for row in rows:
            self._df.write(fmt.format(*[str(v) for v in row]) + "\n")

    # ── convergence summary → diagnostics.log ────────────────────────────────

    def summarize_outer_step(
        self,
        k: int,
        n_steps: int,
        losses: List[float],
        grad_norms: Optional[List[float]] = None,
        elapsed: Optional[float] = None,
        grad_clip: float = 1.0,
    ) -> None:
        """
        Write a structured convergence report for one outer iteration.

        Called automatically by SBBTS.fit() at the end of each outer step.
        Writes to diagnostics.log only.

        Parameters
        ----------
        k          : 1-based outer step index
        n_steps    : total outer steps
        losses     : list of per-epoch average training losses
        grad_norms : list of per-epoch max gradient norms (pre-clip), optional
        elapsed    : wall time in seconds for this step, optional
        """
        if not losses:
            return

        n       = len(losses)
        mean_l  = statistics.mean(losses)
        median_l = statistics.median(losses)
        min_l   = min(losses)
        max_l   = max(losses)
        std_l   = statistics.stdev(losses) if n > 1 else 0.0

        # Trend over first and last N epochs
        window  = max(1, min(100, n // 5))
        first_w = statistics.mean(losses[:window])
        last_w  = statistics.mean(losses[-window:])
        trend   = last_w - first_w

        # Spike detection: counts epochs where loss > 3× mean or 10× minimum
        spike_thresh = max(3.0 * mean_l, min_l * 10.0)
        n_spikes = sum(1 for l in losses if l > spike_thresh)

        # Convergence verdict
        rel = trend / (mean_l + 1e-12)
        if n_spikes > n * 0.05:
            verdict = "UNSTABLE  (>5% spike epochs)"
        elif rel < -0.15:
            verdict = "CONVERGING"
        elif rel < -0.03:
            verdict = "SLOW DESCENT"
        elif abs(rel) <= 0.03:
            verdict = "FLAT (no progress)"
        else:
            verdict = "DIVERGING"

        elapsed_str = f"{elapsed / 60:.1f} min ({elapsed:.0f} s)" if elapsed else "N/A"

        self.section(f"Outer Step k={k}/{n_steps} — Convergence Report")
        self.write(f"  wall_time  : {elapsed_str}")
        self.write(f"  epochs     : {n}")
        self.write(f"  loss stats : mean={mean_l:.4f}  median={median_l:.4f}  "
                   f"min={min_l:.4f}  max={max_l:.4f}  std={std_l:.4f}")
        self.write(f"  trend      : first-{window} avg={first_w:.4f}  "
                   f"last-{window} avg={last_w:.4f}  delta={trend:+.4f}")
        self.write(f"  spikes (>{spike_thresh:.1f}): {n_spikes}  ({100*n_spikes/n:.1f}% of epochs)")
        self.write(f"  verdict    : {verdict}")

        if grad_norms:
            gn = [g for g in grad_norms if g is not None and g == g]
            if gn:
                clipped = sum(1 for g in gn if g > grad_clip)
                clip_label = f"{grad_clip:.1f}" if grad_clip > 0 else "disabled"
                self.write(f"  grad norms : mean={statistics.mean(gn):.4f}  "
                           f"max={max(gn):.4f}  "
                           f"clipped(>{clip_label})={clipped}/{len(gn)} ({100*clipped/len(gn):.1f}%)")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        for f in (self._ef, self._df):
            if not f.closed:
                f.write("\n[END OF LOG]\n")
                f.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"SBBTSLogger(run_dir={self.run_dir})"
