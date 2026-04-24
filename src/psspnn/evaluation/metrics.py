"""
Prediction quality metrics for secondary structure prediction.

Implements the quality indices used in Holley & Karplus (1989) Table 1:

    - Percent correct (three-state total)
    - Matthews correlation coefficient (Matthews 1975) for each state
    - PC(S): percent of predicted-S residues that are correct (precision)

References
----------
Matthews, B. (1975) Biochim. Biophys. Acta 405, 442-451.
Kabsch, W. & Sander, C. (1983) FEBS Lett. 155, 179-182.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

State = Literal["H", "E", "C"]
ALL_STATES: tuple[State, ...] = ("H", "E", "C")


def _confusion_binary(
    predicted: list[str],
    actual: list[str],
    positive: State,
) -> tuple[int, int, int, int]:
    """
    Compute (TP, FP, TN, FN) treating ``positive`` as the positive class.
    All other states are treated as the negative class.
    """
    tp = fp = tn = fn = 0
    for p, a in zip(predicted, actual):
        if p == positive and a == positive:
            tp += 1
        elif p == positive and a != positive:
            fp += 1
        elif p != positive and a == positive:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def matthews_cc(
    predicted: list[str],
    actual: list[str],
    state: State,
) -> float:
    """
    Matthews correlation coefficient for a single secondary structure state.

    C = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Returns 0.0 when the denominator is zero (degenerate predictor).

    Parameters
    ----------
    predicted : list[str]
    actual    : list[str]
    state     : 'H', 'E', or 'C'
    """
    tp, fp, tn, fn = _confusion_binary(predicted, actual, state)
    denom = np.sqrt(
        float((tp + fp)) * float((tp + fn)) * float((tn + fp)) * float((tn + fn))
    )
    if denom == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def percent_correct(predicted: list[str], actual: list[str]) -> float:
    """
    Three-state total percent correct.

    Parameters
    ----------
    predicted : list[str]
    actual    : list[str]

    Returns
    -------
    float in [0, 100].
    """
    n = len(predicted)
    if n == 0:
        return 0.0
    return 100.0 * sum(p == a for p, a in zip(predicted, actual)) / n


def pc_state(
    predicted: list[str],
    actual: list[str],
    state: State,
) -> float:
    """
    PC(S): percent of residues predicted to be in state S that are correct.

    This is the precision for the given state, and is the quality index
    described in Kabsch & Sander (1983) / used in Holley & Karplus Table 1.

    Returns 0.0 if no residues are predicted to be in state S.

    Parameters
    ----------
    predicted : list[str]
    actual    : list[str]
    state     : 'H', 'E', or 'C'
    """
    positives = [(p, a) for p, a in zip(predicted, actual) if p == state]
    if not positives:
        return 0.0
    correct = sum(1 for p, a in positives if a == state)
    return 100.0 * correct / len(positives)


def full_evaluation(
    predicted: list[str],
    actual: list[str],
) -> dict[str, float]:
    """
    Compute all quality indices from Table 1 of Holley & Karplus (1989).

    Parameters
    ----------
    predicted : list[str]
    actual    : list[str]

    Returns
    -------
    dict with keys:
        percent_correct, C_helix, C_sheet, C_coil, PC_helix, PC_sheet, PC_coil
    """
    return {
        "percent_correct": percent_correct(predicted, actual),
        "C_helix": matthews_cc(predicted, actual, "H"),
        "C_sheet": matthews_cc(predicted, actual, "E"),
        "C_coil": matthews_cc(predicted, actual, "C"),
        "PC_helix": pc_state(predicted, actual, "H"),
        "PC_sheet": pc_state(predicted, actual, "E"),
        "PC_coil": pc_state(predicted, actual, "C"),
    }
