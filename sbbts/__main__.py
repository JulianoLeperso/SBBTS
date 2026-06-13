"""
CLI entry point for SBBTS.

Usage:
    python -m sbbts fit       data.npy  --beta 300 --n-steps 5 -o model.pt
    python -m sbbts sample    model.pt  --n 500 -o synthetic.npy
    python -m sbbts augment   model.pt  data.npy --factor 200 -o augmented.npy
    python -m sbbts diagnose  model.pt  data.npy  -o report.png
    python -m sbbts suggest-beta data.npy --T 1.0
"""

import argparse
import sys
from pathlib import Path


def _load(path: str):
    import numpy as np
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".csv":
        return np.loadtxt(p, delimiter=",")
    elif p.suffix == ".npz":
        d = np.load(p)
        return d[list(d.keys())[0]]
    else:
        raise ValueError(f"Unsupported format '{p.suffix}'. Use .npy, .csv, or .npz")


def _ensure_3d(X):
    import numpy as np
    if X.ndim == 2:
        print(f"  Note: treating 2-D array {X.shape} as single trajectory (1, T, d)")
        return X[np.newaxis]
    return X


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_fit(args):
    import numpy as np
    from sbbts import SBBTS

    X = _ensure_3d(_load(args.data))
    N, T_steps, d = X.shape
    print(f"Data: N={N}, T={T_steps}, d={d}")

    beta = args.beta
    if beta is None:
        beta = SBBTS.suggest_beta(T_steps, args.T, safety_factor=5.0)
        print(f"Auto-selected β = {beta:.2f}  (use --beta to override)")

    model = SBBTS(
        beta=beta,
        n_steps=args.n_steps,
        d_model=args.d_model,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
    )
    model.fit(X, T=args.T)

    out = args.output or "model.pt"
    model.save(out)
    print(f"Saved model → {out}")


def cmd_sample(args):
    import numpy as np
    from sbbts import SBBTS

    model = SBBTS.load(args.model)
    X_synth = model.sample(n=args.n)
    out = args.output or "synthetic.npy"
    np.save(out, X_synth)
    print(f"Saved {X_synth.shape} → {out}")


def cmd_augment(args):
    import numpy as np
    from sbbts import SBBTS

    model = SBBTS.load(args.model)
    X = _ensure_3d(_load(args.data))
    X_aug = model.augment(X, factor=args.factor)
    out = args.output or "augmented.npy"
    np.save(out, X_aug)
    print(f"Saved augmented {X_aug.shape} → {out}")


def cmd_diagnose(args):
    import numpy as np
    from sbbts import SBBTS

    model = SBBTS.load(args.model)
    X_real = _ensure_3d(_load(args.data))
    n_synth = min(args.n_synth, len(X_real) * 5)
    print(f"Generating {n_synth} synthetic samples…")
    fig = model.diagnose(X_real, n_synth=n_synth)
    out = args.output or "diagnosis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot → {out}")


def cmd_suggest_beta(args):
    import numpy as np
    from sbbts import SBBTS

    X = _ensure_3d(_load(args.data))
    n_tp = X.shape[1]
    dt = args.T / (n_tp - 1)
    print(f"\nData: T={args.T}, steps={n_tp-1}, Δt={dt:.6f}")
    print(f"Minimum valid β  (β·Δt > 1)  : {1/dt + 0.1:.1f}")
    for sf in [3.0, 5.0, 10.0]:
        print(f"  safety_factor={sf:.0f}  →  β = {SBBTS.suggest_beta(n_tp, args.T, sf):.1f}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        prog="python -m sbbts",
        description="SBBTS – Schrödinger-Bass Bridge for Time Series",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fit
    f = sub.add_parser("fit", help="Fit model to data")
    f.add_argument("data")
    f.add_argument("--beta", type=float, default=None)
    f.add_argument("--n-steps", type=int, default=5)
    f.add_argument("--d-model", type=int, default=128)
    f.add_argument("--n-epochs", type=int, default=1000)
    f.add_argument("--batch-size", type=int, default=128)
    f.add_argument("--T", type=float, default=1.0)
    f.add_argument("-o", "--output", type=str)
    f.add_argument("--early-stopping-patience", type=int, default=50)

    # sample
    s = sub.add_parser("sample", help="Sample from fitted model")
    s.add_argument("model")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("-o", "--output", type=str)

    # augment
    a = sub.add_parser("augment", help="Augment dataset")
    a.add_argument("model")
    a.add_argument("data")
    a.add_argument("--factor", type=int, default=200)
    a.add_argument("-o", "--output", type=str)

    # diagnose
    d = sub.add_parser("diagnose", help="Visual real-vs-synthetic comparison")
    d.add_argument("model")
    d.add_argument("data")
    d.add_argument("--n-synth", type=int, default=500)
    d.add_argument("-o", "--output", type=str)

    # suggest-beta
    sb = sub.add_parser("suggest-beta", help="Suggest β for data")
    sb.add_argument("data")
    sb.add_argument("--T", type=float, default=1.0)

    args = p.parse_args()
    {"fit": cmd_fit, "sample": cmd_sample, "augment": cmd_augment,
     "diagnose": cmd_diagnose, "suggest-beta": cmd_suggest_beta}[args.command](args)


if __name__ == "__main__":
    main()
