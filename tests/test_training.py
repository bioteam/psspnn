"""Tests for psspnn.model.training."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from psspnn.model.network import HolleyKarplusNet
from psspnn.model.training import _sse, train


class TestSSE:
    def test_perfect_prediction(self):
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert _sse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_pred = np.array([[0.0, 0.0]])
        y_true = np.array([[1.0, 1.0]])
        # (0-1)^2 + (0-1)^2 = 2.0
        assert _sse(y_pred, y_true) == pytest.approx(2.0)

    def test_sum_not_mean(self):
        # With 10 samples each contributing 1.0, SSE should be 5.0 not 0.5
        y_pred = np.zeros((10, 2))
        y_true = np.ones((10, 2)) * 0.5
        # each element: (0 - 0.5)^2 = 0.25; 10 * 2 * 0.25 = 5.0
        assert _sse(y_pred, y_true) == pytest.approx(5.0)


class TestTrain:
    def _make_simple_data(self):
        """
        Create a trivially learnable dataset: always predict (1, 0).
        Every sample has the same target so the network can learn quickly.
        """
        rng = np.random.default_rng(0)
        X = rng.random((50, 5 * 21)).astype(np.float32)
        y = np.tile([1.0, 0.0], (50, 1)).astype(np.float32)
        return X, y

    def test_error_decreases(self):
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=0)
        X, y = self._make_simple_data()
        e_init = _sse(net.predict(X), y)
        result = train(net, X, y, learning_rate=0.1, max_cycles=50, verbose=False)
        assert result.final_error < e_init

    def test_error_history_length(self):
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=0)
        X, y = self._make_simple_data()
        result = train(net, X, y, max_cycles=20, verbose=False)
        assert len(result.error_history) == result.n_cycles

    def test_converged_flag_set(self):
        """The converged flag is set when fractional change drops below tol."""
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=42)
        X, y = self._make_simple_data()
        # tol=0.5 fires quickly: the initial rapid drop in error produces
        # frac_change < 0.5 within the first few cycles.
        result = train(
            net,
            X,
            y,
            learning_rate=0.1,
            max_cycles=100,
            tol=0.5,
            verbose=False,
        )
        assert result.converged
        assert result.n_cycles < 100

    def test_max_cycles_respected(self):
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=0)
        X, y = self._make_simple_data()
        result = train(net, X, y, max_cycles=10, tol=1e-20, verbose=False)
        assert result.n_cycles <= 10

    def test_0_hidden_units(self):
        net = HolleyKarplusNet(window_size=5, hidden_units=0, seed=0)
        X, y = self._make_simple_data()
        result = train(net, X, y, max_cycles=50, verbose=False)
        assert result.final_error < _sse(
            np.full_like(y, 0.5), y
        )  # better than constant 0.5 predictor

    def test_training_result_fields(self):
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=0)
        X, y = self._make_simple_data()
        result = train(net, X, y, max_cycles=5, verbose=False)
        assert isinstance(result.n_cycles, int)
        assert isinstance(result.final_error, float)
        assert isinstance(result.error_history, list)
        assert isinstance(result.converged, bool)


class TestCheckpoint:
    def _make_simple_data(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 5 * 21)).astype(np.float32)
        y = np.tile([1.0, 0.0], (50, 1)).astype(np.float32)
        return X, y

    def test_checkpoint_files_created(self):
        """A checkpoint directory is populated when checkpoint_dir is set."""
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=0)
        X, y = self._make_simple_data()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            train(
                net,
                X,
                y,
                max_cycles=5,
                checkpoint_dir=ckpt_dir,
                checkpoint_every=5,
                verbose=False,
            )
            assert ckpt_dir.exists()
            assert (ckpt_dir / "checkpoint.npz").exists()

    def test_resume_continues_from_saved_cycle(self):
        """Resuming from a checkpoint continues where training left off."""
        X, y = self._make_simple_data()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"

            # Phase 1: train for 5 cycles and save a checkpoint
            net1 = HolleyKarplusNet(window_size=5, hidden_units=2, seed=1)
            result1 = train(
                net1,
                X,
                y,
                max_cycles=5,
                tol=1e-20,
                checkpoint_dir=ckpt_dir,
                checkpoint_every=5,
                verbose=False,
            )
            assert result1.n_cycles == 5

            # Phase 2: resume from the checkpoint; should start at cycle 6
            net2 = HolleyKarplusNet(window_size=5, hidden_units=2, seed=1)
            result2 = train(
                net2,
                X,
                y,
                max_cycles=10,
                tol=1e-20,
                checkpoint_dir=ckpt_dir,
                checkpoint_every=10,
                resume=True,
                verbose=False,
            )
            # Only 5 more cycles should have run (6..10)
            assert result2.n_cycles == 10
            assert len(result2.error_history) == 5

    def test_no_checkpoint_when_dir_is_none(self):
        """No checkpoint infrastructure is created when checkpoint_dir is None."""
        net = HolleyKarplusNet(window_size=5, hidden_units=2, seed=0)
        X, y = self._make_simple_data()
        # Should complete without error and produce a normal result
        result = train(net, X, y, max_cycles=5, checkpoint_dir=None, verbose=False)
        assert result.n_cycles == 5
