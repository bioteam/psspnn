"""
Full-batch gradient descent training for the Holley & Karplus network.

Uses tf.GradientTape to implement the Rumelhart et al. (1986)
backpropagation algorithm with the loss and stopping criterion from
Holley & Karplus (1989):

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

import numpy as np
import tensorflow as tf

from .network import HolleyKarplusNet


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
    exact formulation of Eq. 1 in Holley & Karplus (1989).
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
) -> TrainingResult:
    """
    Train the network using full-batch gradient descent via tf.GradientTape.

    Parameters
    ----------
    net           : HolleyKarplusNet  (model weights updated in-place)
    X_train       : (N, n_input)
    y_train       : (N, 2)
    learning_rate : step size for weight updates (default 0.1)
    max_cycles    : hard cap on the number of cycles
    tol           : fractional change threshold for early stopping (2e-4)
    verbose       : print progress
    print_every   : print interval when verbose=True

    Returns
    -------
    TrainingResult with convergence status and error history.

    Notes
    -----
    The loss is the sum of squared errors (not mean). tf.GradientTape
    provides automatic differentiation; the gradients are identical to
    those produced by the Rumelhart et al. (1986) backpropagation
    algorithm because both follow from the chain rule applied to the
    same SSE objective.

    The stopping criterion from the paper is:
        |E_prev - E_curr| / E_prev < tol
    where E is the total SSE over the entire training set.
    """
    X = tf.cast(X_train, tf.float32)
    y = tf.cast(y_train, tf.float32)

    # Plain SGD with no momentum (paper does not mention momentum)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    result = TrainingResult(n_cycles=0, final_error=float("inf"))
    prev_E: float | None = None
    E: float = float("inf")

    for cycle in range(1, max_cycles + 1):
        # Full forward pass + gradient computation over all training samples
        with tf.GradientTape() as tape:
            y_pred = net.model(X, training=True)
            # SSE: sum (not mean) of squared differences — Eq. 1 in paper
            loss = tf.reduce_sum(tf.square(y - y_pred))

        grads = tape.gradient(loss, net.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.model.trainable_variables))

        E = float(loss.numpy())
        result.error_history.append(E)

        # Check stopping criterion
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
