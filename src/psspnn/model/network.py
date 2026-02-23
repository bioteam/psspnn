"""
Neural network for protein secondary structure prediction.

Implements the feed-forward network described in Holley & Karplus (1989)
using TensorFlow/Keras.

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
import tensorflow as tf


class HolleyKarplusNet:
    """
    TensorFlow/Keras implementation of the Holley & Karplus 1989 network.

    Wraps a tf.keras.Sequential model and exposes a numpy-compatible
    predict() method and .npz save/load for persistence.

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
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> tf.keras.Sequential:
        """
        Build and return a Keras Sequential model.

        Weights and biases are initialised uniformly in [-0.1, 0.1] as
        specified by Holley & Karplus (1989).
        """
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        layers: list[tf.keras.layers.Layer] = []

        if self.hidden_units > 0:
            layers.append(
                tf.keras.layers.Dense(
                    self.hidden_units,
                    activation="sigmoid",
                    kernel_initializer=init,
                    bias_initializer=init,
                )
            )
        layers.append(
            tf.keras.layers.Dense(
                2,
                activation="sigmoid",
                kernel_initializer=init,
                bias_initializer=init,
            )
        )

        model = tf.keras.Sequential(layers)
        model.build(input_shape=(None, self.n_input))
        return model

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
        X_tf = tf.cast(X, tf.float32)
        return self.model(X_tf, training=False).numpy()

    # ------------------------------------------------------------------
    # Weight access (for tests and inspection)
    # ------------------------------------------------------------------

    def get_weights(self) -> list[np.ndarray]:
        """Return model weights as a list of numpy arrays."""
        return self.model.get_weights()

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
        weights = self.model.get_weights()
        arrays: dict[str, np.ndarray] = {}
        if self.hidden_units > 0:
            arrays["W1"] = weights[0]   # (n_input, hidden)
            arrays["b1"] = weights[1]   # (hidden,)
            arrays["W2"] = weights[2]   # (hidden, 2)
            arrays["b2"] = weights[3]   # (2,)
        else:
            arrays["W1"] = weights[0]   # (n_input, 2)
            arrays["b1"] = weights[1]   # (2,)
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
        if meta["hidden_units"] > 0:
            net.model.set_weights(
                [data["W1"], data["b1"], data["W2"], data["b2"]]
            )
        else:
            net.model.set_weights([data["W1"], data["b1"]])
        return net
