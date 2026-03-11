"""Tests for psspnn.data.encoding."""

import numpy as np
import pytest

from psspnn.data.encoding import (
    AA_TO_IDX,
    N_AA,
    AMINO_ACIDS,
    SS_TO_TARGET,
    build_dataset,
    encode_null,
    encode_residue,
    sequence_windows,
)


class TestEncodeNull:
    def test_shape(self):
        vec = encode_null()
        assert vec.shape == (N_AA,)

    def test_null_bit_set(self):
        vec = encode_null()
        assert vec[20] == 1.0

    def test_other_bits_zero(self):
        vec = encode_null()
        assert np.sum(vec) == 1.0
        assert vec[:20].sum() == 0.0

    def test_dtype(self):
        assert encode_null().dtype == np.float32


class TestEncodeResidue:
    def test_known_aa(self):
        aa = AMINO_ACIDS[0]          # first amino acid
        vec = encode_residue(aa)
        assert vec[0] == 1.0
        assert np.sum(vec) == 1.0

    def test_alanine_index(self):
        # 'A' should map to index 0 (alphabetical order starts with A)
        vec = encode_residue("A")
        assert vec[AA_TO_IDX["A"]] == 1.0

    def test_case_insensitive(self):
        assert np.array_equal(encode_residue("a"), encode_residue("A"))

    def test_unknown_aa(self):
        vec = encode_residue("X")
        assert vec[20] == 1.0          # null slot
        assert np.sum(vec) == 1.0

    def test_star_maps_to_null(self):
        vec = encode_residue("*")
        assert vec[20] == 1.0

    def test_shape(self):
        assert encode_residue("C").shape == (N_AA,)


class TestSequenceWindows:
    def test_yields_correct_count(self, tiny_sequence, tiny_ss):
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        assert len(windows) == len(tiny_sequence)

    def test_input_shape(self, tiny_sequence, tiny_ss):
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        x, _ = windows[0]
        assert x.shape == (5 * N_AA,)

    def test_target_shape(self, tiny_sequence, tiny_ss):
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        _, y = windows[0]
        assert y.shape == (2,)

    def test_helix_target(self, tiny_sequence, tiny_ss):
        # tiny_ss[0] = 'H' → target should be (0.9, 0.1)
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        _, y = windows[0]
        np.testing.assert_array_almost_equal(y, [0.9, 0.1])

    def test_sheet_target(self, tiny_sequence, tiny_ss):
        # tiny_ss[4] = 'E' → target should be (0.1, 0.9)
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        _, y = windows[4]
        np.testing.assert_array_almost_equal(y, [0.1, 0.9])

    def test_coil_target(self, tiny_sequence, tiny_ss):
        # tiny_ss[6] = 'C' → target should be (0.1, 0.1)
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        _, y = windows[6]
        np.testing.assert_array_almost_equal(y, [0.1, 0.1])

    def test_terminal_padding_uses_null(self, tiny_sequence, tiny_ss):
        # For center=0, window_size=5 (half=2), positions -2 and -1 are padded
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        x, _ = windows[0]
        # First two window positions (null-padded) should have null bit set
        for pos in range(2):
            segment = x[pos * N_AA : (pos + 1) * N_AA]
            assert segment[20] == 1.0

    def test_exactly_one_bit_per_position_interior(self, tiny_sequence, tiny_ss):
        # Interior residue: every window position should have exactly one 1-bit
        windows = list(sequence_windows(tiny_sequence, tiny_ss, window_size=5))
        x, _ = windows[4]  # centre enough to avoid terminal padding
        for pos in range(5):
            segment = x[pos * N_AA : (pos + 1) * N_AA]
            assert segment.sum() == 1.0

    def test_even_window_size_raises(self, tiny_sequence, tiny_ss):
        with pytest.raises(ValueError):
            list(sequence_windows(tiny_sequence, tiny_ss, window_size=4))

    def test_default_window_size(self, tiny_sequence, tiny_ss):
        windows = list(sequence_windows(tiny_sequence, tiny_ss))
        x, _ = windows[0]
        assert x.shape == (17 * N_AA,)


class TestBuildDataset:
    def test_shapes(self, tiny_sequence, tiny_ss):
        proteins = [(tiny_sequence, tiny_ss)]
        X, y = build_dataset(proteins, window_size=5)
        n = len(tiny_sequence)
        assert X.shape == (n, 5 * N_AA)
        assert y.shape == (n, 2)

    def test_two_proteins(self, tiny_sequence, tiny_ss):
        proteins = [(tiny_sequence, tiny_ss), (tiny_sequence, tiny_ss)]
        X, y = build_dataset(proteins, window_size=5)
        assert X.shape[0] == 2 * len(tiny_sequence)

    def test_dtype(self, tiny_sequence, tiny_ss):
        X, y = build_dataset([(tiny_sequence, tiny_ss)], window_size=5)
        assert X.dtype == np.float32
        assert y.dtype == np.float32
