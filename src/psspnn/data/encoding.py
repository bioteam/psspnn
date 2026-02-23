"""
One-hot window encoding for protein sequences.

Implements the binary encoding scheme described in Holley & Karplus (1989):
- Each amino acid at each window position is encoded by 21 bits
  (one per amino acid type + one null for terminal padding)
- For any window, exactly one bit per position is 1
- The input layer is window_size * 21 units (357 for the default window of 17)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

# 20 standard amino acids in alphabetical order (by one-letter code).
# This ordering is internal and must be consistent across the entire codebase.
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NULL_IDX: int = 20  # index for the null/padding token
N_AA: int = 21      # 20 amino acids + 1 null

# Target vectors for the two output units: (helix, sheet)
# (1,0)=helix, (0,1)=sheet, (0,0)=coil
SS_TO_TARGET: dict[str, np.ndarray] = {
    "H": np.array([1.0, 0.0], dtype=np.float32),
    "E": np.array([0.0, 1.0], dtype=np.float32),
    "C": np.array([0.0, 0.0], dtype=np.float32),
}


def encode_residue(aa: str) -> np.ndarray:
    """
    Encode a single amino acid as a 21-bit one-hot vector.

    Non-standard or unknown amino acids are encoded with the null bit set.
    Returns shape (N_AA,) = (21,).
    """
    vec = np.zeros(N_AA, dtype=np.float32)
    idx = AA_TO_IDX.get(aa.upper())
    if idx is not None:
        vec[idx] = 1.0
    else:
        vec[NULL_IDX] = 1.0
    return vec


def encode_null() -> np.ndarray:
    """
    Return the null/padding vector used when the window extends past a
    sequence terminus. Index 20 is set to 1; all others are 0.
    Returns shape (N_AA,) = (21,).
    """
    vec = np.zeros(N_AA, dtype=np.float32)
    vec[NULL_IDX] = 1.0
    return vec


def sequence_windows(
    sequence: str,
    ss_labels: list[str],
    window_size: int = 17,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield (input_vector, target_vector) for every residue in the sequence.

    The window is centred on each residue. Positions that extend past either
    end of the sequence are filled with the null vector (index 20 = 1).

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.
    ss_labels : list[str]
        Per-residue secondary structure labels ('H', 'E', or 'C').
        Must have the same length as sequence.
    window_size : int
        Width of the sliding window. Must be odd. Default 17.

    Yields
    ------
    x : np.ndarray, shape (window_size * N_AA,)
        Flattened one-hot encoding of the window.
    y : np.ndarray, shape (2,)
        Target output vector for the central residue.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd; got {window_size}.")
    half = window_size // 2
    n = len(sequence)
    for center in range(n):
        parts: list[np.ndarray] = []
        for offset in range(-half, half + 1):
            pos = center + offset
            if 0 <= pos < n:
                parts.append(encode_residue(sequence[pos]))
            else:
                parts.append(encode_null())
        x = np.concatenate(parts)  # (window_size * N_AA,)
        y = SS_TO_TARGET[ss_labels[center]]
        yield x, y


def build_dataset(
    proteins: list[tuple[str, list[str]]],
    window_size: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X, y arrays from a list of (sequence, ss_labels) tuples.

    Parameters
    ----------
    proteins : list of (sequence, ss_labels) tuples
    window_size : int

    Returns
    -------
    X : np.ndarray, shape (N_residues_total, window_size * N_AA)
    y : np.ndarray, shape (N_residues_total, 2)
    """
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for seq, ss in proteins:
        for x, y_vec in sequence_windows(seq, ss, window_size):
            all_x.append(x)
            all_y.append(y_vec)
    X = np.array(all_x, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    return X, y
