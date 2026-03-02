"""
Neural network for protein secondary structure prediction.

Implements the feed-forward network described in Holley & Karplus (1989)
using numpy.

Architecture:
    (window_size * 21)  ->  [hidden_units]  ->  2
All units use a sigmoid activation: Y_k = 1 / (1 + exp(-X_k)).

When hidden_units=0 the hidden layer is omitted entirely, producing the
input-to-output network analysed in Table 4 and Table 5 of the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class HolleyKarplusNet:
    """
    Numpy implementation of the Holley & Karplus 1989 network.

    Parameters
    ----------
    window_size : int
        Width of the sliding input window. Default 17.
    hidden_units : int
        Number of units in the hidden layer. Use 0 to remove the hidden
        layer entirely (input connects directly to output). Default 2.
    seed : int or None
        Random seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        window_size: int = 17,
        hidden_units: int = 2,
        seed: int | None = None,
    ) -> None:
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.seed = seed
        self.n_input: int = window_size * 21  # 357 for default window
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Initialise weights uniformly in [-0.1, 0.1] as specified by
        Holley & Karplus (1989).
        """
        rng = np.random.default_rng(self.seed)
        if self.hidden_units > 0:
            self.W1 = rng.uniform(-0.1, 0.1, (self.n_input, self.hidden_units)).astype(np.float32)
            self.b1 = rng.uniform(-0.1, 0.1, (self.hidden_units,)).astype(np.float32)
            self.W2 = rng.uniform(-0.1, 0.1, (self.hidden_units, 2)).astype(np.float32)
            self.b2 = rng.uniform(-0.1, 0.1, (2,)).astype(np.float32)
        else:
            self.W1 = rng.uniform(-0.1, 0.1, (self.n_input, 2)).astype(np.float32)
            self.b1 = rng.uniform(-0.1, 0.1, (2,)).astype(np.float32)
            self.W2 = None
            self.b2 = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw output activations as a numpy array.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_input)

        Returns
        -------
        np.ndarray, shape (N, 2), values in (0, 1).
        """
        X = np.asarray(X, dtype=np.float32)
        if self.hidden_units > 0:
            h = _sigmoid(X @ self.W1 + self.b1)
            return _sigmoid(h @ self.W2 + self.b2)
        else:
            return _sigmoid(X @ self.W1 + self.b1)

    # ------------------------------------------------------------------
    # Weight access (for tests and inspection)
    # ------------------------------------------------------------------

    def get_weights(self) -> list[np.ndarray]:
        """Return model weights as a list of numpy arrays."""
        if self.hidden_units > 0:
            return [self.W1, self.b1, self.W2, self.b2]
        return [self.W1, self.b1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save weights to path.npz and hyperparameters to path.json.

        Parameters
        ----------
        path : Path
            Should have a .npz extension. A sidecar .json file is written
            at the same location with the extension replaced by .json.
        """
        path = Path(path)
        arrays: dict[str, np.ndarray] = {"W1": self.W1, "b1": self.b1}
        if self.hidden_units > 0:
            arrays["W2"] = self.W2
            arrays["b2"] = self.b2
        np.savez(path, **arrays)
        meta = {
            "window_size": self.window_size,
            "hidden_units": self.hidden_units,
            "seed": self.seed,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "HolleyKarplusNet":
        """
        Load weights saved by save().

        Parameters
        ----------
        path : Path
            Path to the .npz weights file. The sidecar .json must exist
            at the same location.
        """
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        net = cls(
            window_size=meta["window_size"],
            hidden_units=meta["hidden_units"],
            seed=meta.get("seed"),
        )
        data = np.load(path)
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        if meta["hidden_units"] > 0:
            net.W2 = data["W2"]
            net.b2 = data["b2"]
        return net
