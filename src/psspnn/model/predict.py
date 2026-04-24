"""
Post-processing of raw network output into H/E/C predictions.

Implements the threshold and contiguity rules from Holley & Karplus (1989):

    "Helix is assigned to any group of 4 or more contiguous residues,
     the minimum helix in Kabsch and Sander classifications, having helix
     output values greater than sheet outputs and greater than t."

    "Beta-Strand is assigned to any group of 2 or more contiguous residues,
     the minimum beta-strand, having sheet output values greater than helix
     outputs and greater than t."

    "Residues not assigned to helices or beta-strands are considered coil."

The threshold t=0.37 was selected by maximising training-set accuracy.
"""

from __future__ import annotations

import numpy as np


def raw_state(
    helix_out: np.ndarray,
    sheet_out: np.ndarray,
    threshold: float = 0.37,
) -> list[str]:
    """
    Assign a raw (pre-contiguity) secondary structure state to each residue.

    Parameters
    ----------
    helix_out : np.ndarray, shape (N,)
        Network output unit 0 activations (helix signal).
    sheet_out : np.ndarray, shape (N,)
        Network output unit 1 activations (sheet signal).
    threshold : float
        Minimum value the winning output must exceed. Default 0.37.

    Returns
    -------
    list[str]
        Per-residue labels 'H', 'E', or 'C' before contiguity filtering.
    """
    states: list[str] = []
    for h, s in zip(helix_out, sheet_out):
        if h > s and h > threshold:
            states.append("H")
        elif s > h and s > threshold:
            states.append("E")
        else:
            states.append("C")
    return states


def apply_contiguity(
    raw: list[str],
    min_helix: int = 4,
    min_sheet: int = 2,
) -> list[str]:
    """
    Apply minimum-length rules from Kabsch & Sander classifications.

    Helix and sheet runs shorter than the minimum length are converted to coil.

    Parameters
    ----------
    raw       : list[str]  — per-residue labels ('H', 'E', 'C')
    min_helix : int        — minimum helix run length (default 4)
    min_sheet : int        — minimum beta-strand run length (default 2)

    Returns
    -------
    list[str]  — filtered labels
    """
    result = list(raw)
    for state, min_len in [("H", min_helix), ("E", min_sheet)]:
        i = 0
        while i < len(result):
            if result[i] == state:
                j = i + 1
                while j < len(result) and result[j] == state:
                    j += 1
                # result[i:j] is a run of `state`
                if j - i < min_len:
                    for k in range(i, j):
                        result[k] = "C"
                i = j
            else:
                i += 1
    return result


def predict_sequence(
    net,
    sequence: str,
    window_size: int = 17,
    threshold: float = 0.37,
) -> tuple[np.ndarray, list[str]]:
    """
    Full prediction pipeline for a single amino acid sequence.

    Parameters
    ----------
    net         : HolleyKarplusNet instance
    sequence    : str  — one-letter amino acid sequence
    window_size : int
    threshold   : float

    Returns
    -------
    raw_outputs : np.ndarray, shape (N, 2)
        Raw network activations for every residue (before thresholding).
    final_preds : list[str]
        Per-residue labels after thresholding and contiguity filtering.
    """
    from ..data.encoding import sequence_windows

    dummy_ss = ["C"] * len(sequence)
    windows = [x for x, _ in sequence_windows(sequence, dummy_ss, window_size)]
    X = np.array(windows, dtype=np.float64)
    raw_out = net.predict(X)  # (N, 2)
    raw = raw_state(raw_out[:, 0], raw_out[:, 1], threshold)
    final = apply_contiguity(raw)
    return raw_out, final
