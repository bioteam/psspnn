"""Tests for psspnn.evaluation.metrics."""

import pytest

from psspnn.evaluation.metrics import (
    full_evaluation,
    matthews_cc,
    pc_state,
    percent_correct,
)


class TestPercentCorrect:
    def test_perfect(self):
        pred = ["H", "H", "E", "C"]
        actual = ["H", "H", "E", "C"]
        assert percent_correct(pred, actual) == pytest.approx(100.0)

    def test_all_wrong(self):
        pred = ["H", "H", "H"]
        actual = ["C", "C", "C"]
        assert percent_correct(pred, actual) == pytest.approx(0.0)

    def test_two_thirds_correct(self):
        pred = ["H", "H", "E"]
        actual = ["H", "E", "E"]
        assert percent_correct(pred, actual) == pytest.approx(100 * 2 / 3)

    def test_empty(self):
        assert percent_correct([], []) == pytest.approx(0.0)


class TestMatthewsCC:
    def test_perfect_helix_predictor(self):
        # All predictions correct
        pred   = ["H", "H", "C", "C", "E", "E"]
        actual = ["H", "H", "C", "C", "E", "E"]
        cc = matthews_cc(pred, actual, "H")
        assert cc == pytest.approx(1.0)

    def test_all_predicted_negative(self):
        # Never predict H → TP=0, FP=0 → denominator might be 0
        pred   = ["C", "C", "E"]
        actual = ["H", "C", "E"]
        cc = matthews_cc(pred, actual, "H")
        assert cc == pytest.approx(0.0)

    def test_range(self):
        import random
        rng = random.Random(0)
        states = ["H", "E", "C"]
        pred   = [rng.choice(states) for _ in range(100)]
        actual = [rng.choice(states) for _ in range(100)]
        cc = matthews_cc(pred, actual, "H")
        assert -1.0 <= cc <= 1.0

    def test_perfectly_wrong(self):
        # Predict H for everything that is non-H, and non-H for H
        pred   = ["C", "C", "H", "H"]
        actual = ["H", "H", "C", "C"]
        cc = matthews_cc(pred, actual, "H")
        assert cc == pytest.approx(-1.0)

    def test_sheet_state(self):
        pred   = ["E", "E", "H", "C"]
        actual = ["E", "H", "H", "C"]
        cc = matthews_cc(pred, actual, "E")
        # TP=1, FP=1, TN=2, FN=0
        # C = (1*2 - 1*0) / sqrt(2*1*3*2) = 2 / sqrt(12)
        expected = (1 * 2 - 1 * 0) / (2 * 1 * 3 * 2) ** 0.5
        assert cc == pytest.approx(expected)


class TestPCState:
    def test_perfect_precision(self):
        pred   = ["H", "H", "C"]
        actual = ["H", "H", "C"]
        assert pc_state(pred, actual, "H") == pytest.approx(100.0)

    def test_zero_precision(self):
        pred   = ["H", "H"]
        actual = ["C", "C"]
        assert pc_state(pred, actual, "H") == pytest.approx(0.0)

    def test_half_precision(self):
        pred   = ["H", "H"]
        actual = ["H", "C"]
        assert pc_state(pred, actual, "H") == pytest.approx(50.0)

    def test_no_predictions_of_state(self):
        pred   = ["C", "C"]
        actual = ["H", "C"]
        assert pc_state(pred, actual, "H") == pytest.approx(0.0)


class TestFullEvaluation:
    def test_keys(self):
        pred   = ["H", "E", "C"]
        actual = ["H", "E", "C"]
        result = full_evaluation(pred, actual)
        expected_keys = {
            "percent_correct", "C_helix", "C_sheet", "C_coil",
            "PC_helix", "PC_sheet", "PC_coil",
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_prediction(self):
        pred   = ["H", "H", "E", "E", "C", "C"]
        actual = ["H", "H", "E", "E", "C", "C"]
        result = full_evaluation(pred, actual)
        assert result["percent_correct"] == pytest.approx(100.0)
        assert result["C_helix"]  == pytest.approx(1.0)
        assert result["C_sheet"]  == pytest.approx(1.0)
        assert result["C_coil"]   == pytest.approx(1.0)
        assert result["PC_helix"] == pytest.approx(100.0)
        assert result["PC_sheet"] == pytest.approx(100.0)
        assert result["PC_coil"]  == pytest.approx(100.0)
