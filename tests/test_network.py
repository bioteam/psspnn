"""Tests for psspnn.model.network."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from psspnn.model.network import HolleyKarplusNet


class TestInit:
    def test_default_shapes_2h(self):
        net = HolleyKarplusNet(window_size=17, hidden_units=2)
        W1, b1, W2, b2 = net.get_weights()
        assert W1.shape == (357, 2)
        assert b1.shape == (2,)
        assert W2.shape == (2, 2)
        assert b2.shape == (2,)

    def test_default_shapes_0h(self):
        net = HolleyKarplusNet(window_size=17, hidden_units=0)
        W1, b1 = net.get_weights()
        assert W1.shape == (357, 2)
        assert b1.shape == (2,)

    def test_weight_range(self):
        net = HolleyKarplusNet(window_size=17, hidden_units=2, seed=0)
        W1, b1, W2, b2 = net.get_weights()
        assert W1.min() >= -0.1 - 1e-6
        assert W1.max() <= 0.1 + 1e-6
        assert b1.min() >= -0.1 - 1e-6
        assert b1.max() <= 0.1 + 1e-6

    def test_small_window(self):
        net = HolleyKarplusNet(window_size=5, hidden_units=2)
        W1, b1, W2, b2 = net.get_weights()
        assert W1.shape == (5 * 21, 2)

    def test_5_hidden_units(self):
        net = HolleyKarplusNet(window_size=17, hidden_units=5)
        W1, b1, W2, b2 = net.get_weights()
        assert W1.shape == (357, 5)
        assert W2.shape == (5, 2)

    def test_reproducibility_with_seed(self):
        net1 = HolleyKarplusNet(window_size=5, hidden_units=2, seed=7)
        net2 = HolleyKarplusNet(window_size=5, hidden_units=2, seed=7)
        W1_a = net1.get_weights()[0]
        W1_b = net2.get_weights()[0]
        np.testing.assert_array_equal(W1_a, W1_b)

    def test_different_seeds_produce_different_weights(self):
        net1 = HolleyKarplusNet(window_size=5, hidden_units=2, seed=1)
        net2 = HolleyKarplusNet(window_size=5, hidden_units=2, seed=2)
        assert not np.array_equal(net1.get_weights()[0], net2.get_weights()[0])


class TestPredict:
    def test_output_range_2h(self, net_2h):
        X = np.random.default_rng(0).random((20, net_2h.n_input)).astype(np.float32)
        Y = net_2h.predict(X)
        assert Y.shape == (20, 2)
        assert Y.min() > 0.0
        assert Y.max() < 1.0

    def test_output_range_0h(self, net_0h):
        X = np.random.default_rng(0).random((20, net_0h.n_input)).astype(np.float32)
        Y = net_0h.predict(X)
        assert Y.shape == (20, 2)
        assert Y.min() > 0.0
        assert Y.max() < 1.0

    def test_single_sample(self, net_2h):
        X = np.zeros((1, net_2h.n_input), dtype=np.float32)
        Y = net_2h.predict(X)
        assert Y.shape == (1, 2)

    def test_returns_numpy(self, net_2h):
        X = np.random.default_rng(0).random((5, net_2h.n_input)).astype(np.float32)
        Y = net_2h.predict(X)
        assert isinstance(Y, np.ndarray)

    def test_deterministic(self, net_2h):
        X = np.random.default_rng(0).random((5, net_2h.n_input)).astype(np.float32)
        Y1 = net_2h.predict(X)
        Y2 = net_2h.predict(X)
        np.testing.assert_array_equal(Y1, Y2)


class TestSaveLoad:
    def test_roundtrip_2h(self, net_2h):
        X = np.random.default_rng(0).random((5, net_2h.n_input)).astype(np.float32)
        before = net_2h.predict(X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            net_2h.save(path)
            loaded = HolleyKarplusNet.load(path)
        after = loaded.predict(X)
        np.testing.assert_array_almost_equal(before, after)

    def test_roundtrip_0h(self, net_0h):
        X = np.random.default_rng(0).random((5, net_0h.n_input)).astype(np.float32)
        before = net_0h.predict(X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            net_0h.save(path)
            loaded = HolleyKarplusNet.load(path)
        after = loaded.predict(X)
        np.testing.assert_array_almost_equal(before, after)

    def test_json_sidecar_created(self, net_2h):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            net_2h.save(path)
            assert path.with_suffix(".json").exists()

    def test_hyperparameters_preserved(self, net_2h):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            net_2h.save(path)
            loaded = HolleyKarplusNet.load(path)
        assert loaded.window_size == net_2h.window_size
        assert loaded.hidden_units == net_2h.hidden_units
