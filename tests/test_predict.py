"""Tests for psspnn.model.predict."""

import numpy as np
import pytest

from psspnn.data.encoding import build_dataset
from psspnn.model.network import HolleyKarplusNet
from psspnn.model.predict import apply_contiguity, predict_sequence, raw_state
from psspnn.model.training import train


class TestRawState:
    def test_helix(self):
        assert raw_state(np.array([0.8]), np.array([0.2])) == ["H"]

    def test_sheet(self):
        assert raw_state(np.array([0.2]), np.array([0.8])) == ["E"]

    def test_coil_neither_exceeds_threshold(self):
        # Neither exceeds threshold 0.37
        assert raw_state(np.array([0.3]), np.array([0.2])) == ["C"]

    def test_coil_below_threshold_even_if_dominant(self):
        # h > s but h < threshold → coil
        assert raw_state(np.array([0.36]), np.array([0.1]), threshold=0.37) == ["C"]

    def test_tie_goes_to_coil(self):
        # h == s → neither wins → coil
        assert raw_state(np.array([0.5]), np.array([0.5])) == ["C"]

    def test_batch(self):
        h = np.array([0.8, 0.2, 0.3])
        s = np.array([0.2, 0.8, 0.2])
        assert raw_state(h, s) == ["H", "E", "C"]

    def test_custom_threshold(self):
        # With threshold=0.5, 0.45 should be coil
        assert raw_state(np.array([0.45]), np.array([0.1]), threshold=0.5) == ["C"]
        # With threshold=0.4, 0.45 should be helix
        assert raw_state(np.array([0.45]), np.array([0.1]), threshold=0.4) == ["H"]


class TestApplyContiguity:
    def test_short_helix_removed(self):
        raw = ["H", "H", "H", "C", "C"]  # 3-run < 4
        result = apply_contiguity(raw, min_helix=4, min_sheet=2)
        assert result == ["C", "C", "C", "C", "C"]

    def test_long_helix_kept(self):
        raw = ["H", "H", "H", "H", "C"]  # 4-run == 4
        result = apply_contiguity(raw, min_helix=4)
        assert result == ["H", "H", "H", "H", "C"]

    def test_short_sheet_removed(self):
        raw = ["E", "C", "C"]  # 1-run < 2
        result = apply_contiguity(raw, min_sheet=2)
        assert result == ["C", "C", "C"]

    def test_min_sheet_run_kept(self):
        raw = ["E", "E", "C"]  # 2-run == 2
        result = apply_contiguity(raw, min_sheet=2)
        assert result == ["E", "E", "C"]

    def test_all_coil_unchanged(self):
        raw = ["C", "C", "C"]
        assert apply_contiguity(raw) == raw

    def test_mixed(self):
        raw = ["H", "H", "H", "C", "E", "E", "C"]
        result = apply_contiguity(raw, min_helix=4, min_sheet=2)
        # H-run of 3 → C; E-run of 2 stays
        assert result == ["C", "C", "C", "C", "E", "E", "C"]

    def test_output_length_unchanged(self):
        raw = ["H"] * 10 + ["E"] * 3 + ["C"] * 5
        result = apply_contiguity(raw)
        assert len(result) == len(raw)

    def test_input_not_mutated(self):
        raw = ["H", "H", "C"]
        original = list(raw)
        apply_contiguity(raw)
        assert raw == original


class TestPredictSequence:
    def test_output_shapes(self, net_2h, tiny_sequence):
        raw_out, final_preds = predict_sequence(net_2h, tiny_sequence, window_size=5)
        assert raw_out.shape == (len(tiny_sequence), 2)
        assert len(final_preds) == len(tiny_sequence)

    def test_outputs_in_0_1(self, net_2h, tiny_sequence):
        raw_out, _ = predict_sequence(net_2h, tiny_sequence, window_size=5)
        assert raw_out.min() > 0.0
        assert raw_out.max() < 1.0

    def test_predictions_are_valid_states(self, net_2h, tiny_sequence):
        _, final_preds = predict_sequence(net_2h, tiny_sequence, window_size=5)
        assert all(s in {"H", "E", "C"} for s in final_preds)

    def test_single_residue(self, net_2h):
        raw_out, preds = predict_sequence(net_2h, "A", window_size=5)
        assert raw_out.shape == (1, 2)
        assert len(preds) == 1


class TestEndToEnd:
    """Train a small network and verify it predicts known secondary structures."""

    @pytest.fixture(scope="class")
    def trained_net(self):
        """Train a network on synthetic proteins with known structure."""
        helix_seq = "AEAAAKEAAAKEAAAKEAAAK"
        helix_ss = ["H"] * len(helix_seq)

        sheet_seq = "VVIVTIVVIVTIVVIVTIVVV"
        sheet_ss = ["E"] * len(sheet_seq)

        coil_seq = "GSNGSGSNGSGSNGSGSNGSG"
        coil_ss = ["C"] * len(coil_seq)

        proteins = [
            (helix_seq, helix_ss),
            (sheet_seq, sheet_ss),
            (coil_seq, coil_ss),
        ]
        X, y = build_dataset(proteins, window_size=17)
        net = HolleyKarplusNet(window_size=17, hidden_units=2, seed=42)
        train(net, X, y, learning_rate=0.1, max_cycles=2000, tol=1e-6)
        return net

    def test_helix_prediction(self, trained_net):
        """A strongly helical sequence should be predicted mostly as helix."""
        _, preds = predict_sequence(trained_net, "AEAAAKEAAAKEAAAKEAAAK")
        helix_frac = preds.count("H") / len(preds)
        assert helix_frac > 0.5, f"Expected mostly helix, got {helix_frac:.0%} H"

    def test_sheet_prediction(self, trained_net):
        """A strongly sheet-like sequence should be predicted mostly as sheet."""
        _, preds = predict_sequence(trained_net, "VVIVTIVVIVTIVVIVTIVVV")
        sheet_frac = preds.count("E") / len(preds)
        assert sheet_frac > 0.5, f"Expected mostly sheet, got {sheet_frac:.0%} E"

    def test_coil_prediction(self, trained_net):
        """A coil-like sequence should be predicted mostly as coil."""
        _, preds = predict_sequence(trained_net, "GSNGSGSNGSGSNGSGSNGSG")
        coil_frac = preds.count("C") / len(preds)
        assert coil_frac > 0.5, f"Expected mostly coil, got {coil_frac:.0%} C"


class TestEndToEndMixed:
    """Train on a mixed H/E/C protein and verify all three states are predicted."""

    # Sperm whale myoglobin B-E helix region with flanking coil and strand
    MYO_SEQ = "LFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILK"
    #          CHHHHCHHHHHHCCCCHHHHHHHHHHHHHHHCCCEEECC HHHHHHHH
    MYO_SS = list(
        "CHHHHCHHHHHHCCCCHHHHHHHHHHHHHHCCCEEECCHHHHHHHH"
    )

    @pytest.fixture(scope="class")
    def trained_net(self):
        helix_seq = "AEAAAKEAAAKEAAAKEAAAK"
        sheet_seq = "VVIVTIVVIVTIVVIVTIVVV"
        coil_seq = "GSNGSGSNGSGSNGSGSNGSG"
        proteins = [
            (helix_seq, ["H"] * len(helix_seq)),
            (sheet_seq, ["E"] * len(sheet_seq)),
            (coil_seq, ["C"] * len(coil_seq)),
            (self.MYO_SEQ, self.MYO_SS),
        ]
        X, y = build_dataset(proteins, window_size=17)
        net = HolleyKarplusNet(window_size=17, hidden_units=2, seed=42)
        train(net, X, y, learning_rate=0.1, max_cycles=2000, tol=1e-6)
        return net

    def test_all_three_states_predicted(self, trained_net):
        """A mixed-structure sequence should produce H, E, and C predictions."""
        _, preds = predict_sequence(trained_net, self.MYO_SEQ)
        states = set(preds)
        assert "H" in states, "Expected helix predictions"
        assert "E" in states, "Expected sheet predictions"
        assert "C" in states, "Expected coil predictions"

    def test_helix_regions_mostly_correct(self, trained_net):
        """Residues known to be helix should be predicted mostly as helix."""
        _, preds = predict_sequence(trained_net, self.MYO_SEQ)
        helix_indices = [i for i, s in enumerate(self.MYO_SS) if s == "H"]
        helix_correct = sum(1 for i in helix_indices if preds[i] == "H")
        frac = helix_correct / len(helix_indices)
        assert frac > 0.5, f"Expected >50% helix correct, got {frac:.0%}"

    def test_coil_regions_mostly_correct(self, trained_net):
        """Residues known to be coil should be predicted mostly as coil."""
        _, preds = predict_sequence(trained_net, self.MYO_SEQ)
        coil_indices = [i for i, s in enumerate(self.MYO_SS) if s == "C"]
        coil_correct = sum(1 for i in coil_indices if preds[i] == "C")
        frac = coil_correct / len(coil_indices)
        assert frac > 0.3, f"Expected >30% coil correct, got {frac:.0%}"
