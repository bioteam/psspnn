"""
Full-batch gradient descent training for the Holley & Karplus network.

Implements the Rumelhart et al. (1986) backpropagation algorithm with the
loss and stopping criterion from Holley & Karplus (1989):

    Loss:  E = sum_c sum_j (O_{j,c} - D_{j,c})^2      [Eq. 1 in paper]
    Stop:  |E_prev - E_curr| / E_prev < 2e-4

One "cycle" = one complete pass through all training samples with a
single weight update at the end. This matches the paper: "all of the
training proteins are presented to the network and the input window
moves through the amino acid sequences one residue at a time. At the
end of the cycle, weights are adjusted."

The learning rate is not stated in the paper; 0.1 is the default from
Rumelhart et al. (1986).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .network import HolleyKarplusNet, _sigmoid


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Summary statistics returned by train()."""
    n_cycles: int
    final_error: float
    error_history: list[float] = field(default_factory=list)
    converged: bool = False


# ---------------------------------------------------------------------------
# Loss (kept as a standalone function for testability)
# ---------------------------------------------------------------------------

def _sse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Sum of squared errors: E = sum_c sum_j (O_{j,c} - D_{j,c})^2.

    Neither averaged over samples nor over output units — this is the
    formulation of Eq. 1 in Holley & Karplus (1989).
    """
    return float(np.sum((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    net: HolleyKarplusNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.1,
    max_cycles: int = 2000,
    tol: float = 2e-4,
    verbose: bool = True,
    print_every: int = 10,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 100,
    resume: bool = False,
) -> TrainingResult:
    """
    Train the network using full-batch gradient descent with manual backpropagation

    Parameters
    ----------
    net              : HolleyKarplusNet  (model weights updated in-place)
    X_train          : (N, n_input)
    y_train          : (N, 2)
    learning_rate    : step size for weight updates (default 0.1)
    max_cycles       : hard cap on the number of cycles
    tol              : fractional change threshold for early stopping (2e-4)
    verbose          : print progress
    print_every      : print interval when verbose=True
    checkpoint_dir   : directory for periodic checkpoint saves; no checkpoints
                       are written when None (default)
    checkpoint_every : save a checkpoint every N cycles (default 100)
    resume           : if True and checkpoint_dir contains a checkpoint, restore
                       model weights and continue from that cycle

    Returns
    -------
    TrainingResult with convergence status and error history.

    Notes
    -----
    The loss is the sum of squared errors (not mean). Gradients are computed
    via the Rumelhart et al. (1986) backpropagation algorithm applied to
    the SSE objective with sigmoid activations.

    The stopping criterion from the paper is:
        |E_prev - E_curr| / E_prev < tol
    where E is the total SSE over the entire training set.

    Checkpointing saves model weights and cycle number to an .npz file,
    allowing resumption after an interruption.
    """
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    start_cycle = 1

    # ------------------------------------------------------------------
    # Checkpoint restore
    # ------------------------------------------------------------------
    if checkpoint_dir is not None and resume:
        ckpt_path = Path(checkpoint_dir)
        ckpt_file = ckpt_path / "checkpoint.npz"
        if ckpt_file.exists():
            data = np.load(ckpt_file)
            net.W1 = data["W1"]
            net.b1 = data["b1"]
            if net.hidden_units > 0:
                net.W2 = data["W2"]
                net.b2 = data["b2"]
            start_cycle = int(data["cycle"]) + 1
            if verbose:
                print(f"Resumed from checkpoint (cycle {start_cycle - 1})")
        elif verbose:
            print("No checkpoint found; starting from cycle 1.")

    result = TrainingResult(n_cycles=0, final_error=float("inf"))
    prev_E: float | None = None
    E: float = float("inf")

    for cycle in range(start_cycle, max_cycles + 1):
        # ----------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------
        if net.hidden_units > 0:
            z_h = X @ net.W1 + net.b1            # (N, hidden)
            h = _sigmoid(z_h)                    # (N, hidden)
            z_o = h @ net.W2 + net.b2            # (N, 2)
            out = _sigmoid(z_o)                  # (N, 2)
        else:
            z_o = X @ net.W1 + net.b1            # (N, 2)
            out = _sigmoid(z_o)                  # (N, 2)

        # ----------------------------------------------------------
        # Loss
        # ----------------------------------------------------------
        E = _sse(out, y)
        result.error_history.append(E)

        # ----------------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------------
        # dE/do_j = 2 * (o_j - y_j)
        # delta_out = dE/do * sigmoid'(z_o) = 2*(out - y) * out*(1-out)
        delta_out = 2.0 * (out - y) * out * (1.0 - out)  # (N, 2)

        if net.hidden_units > 0:
            dW2 = h.T @ delta_out                 # (hidden, 2)
            db2 = delta_out.sum(axis=0)           # (2,)
            delta_h = (delta_out @ net.W2.T) * h * (1.0 - h)  # (N, hidden)
            dW1 = X.T @ delta_h                   # (n_input, hidden)
            db1 = delta_h.sum(axis=0)             # (hidden,)
        else:
            dW1 = X.T @ delta_out                 # (n_input, 2)
            db1 = delta_out.sum(axis=0)           # (2,)

        # ----------------------------------------------------------
        # Weight update (plain SGD, no momentum)
        # ----------------------------------------------------------
        net.W1 -= learning_rate * dW1
        net.b1 -= learning_rate * db1
        if net.hidden_units > 0:
            net.W2 -= learning_rate * dW2
            net.b2 -= learning_rate * db2

        # ----------------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------------
        if checkpoint_dir is not None and cycle % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, np.ndarray] = {
                "W1": net.W1, "b1": net.b1, "cycle": np.array(cycle),
            }
            if net.hidden_units > 0:
                arrays["W2"] = net.W2
                arrays["b2"] = net.b2
            np.savez(ckpt_path / "checkpoint.npz", **arrays)
            if verbose:
                print(f"  Checkpoint saved at cycle {cycle}")

        # ----------------------------------------------------------
        # Stopping criterion
        # ----------------------------------------------------------
        if prev_E is not None:
            frac_change = abs(prev_E - E) / (abs(prev_E) + 1e-12)
            if verbose and cycle % print_every == 0:
                print(f"Cycle {cycle:5d}  E={E:.4f}  Delta={frac_change:.2e}")
            if frac_change < tol:
                if verbose:
                    print(
                        f"Converged at cycle {cycle}  "
                        f"E={E:.4f}  Delta={frac_change:.2e}"
                    )
                result.converged = True
                break
        elif verbose:
            print(f"Cycle {cycle:5d}  E={E:.4f}")

        prev_E = E

    result.n_cycles = cycle
    result.final_error = E
    return result


