"""
DSSP secondary structure assignment using pydssp.

Parses PDB files with biopython, extracts backbone coordinates, then
calls pydssp.assign() to obtain per-residue DSSP codes, which are mapped
to the 3-state scheme (H, E, C) used by Holley & Karplus (1989).

The 3-state mapping follows Kabsch & Sander (1983):
    H, G, I  →  H  (alpha-helix, 3-10 helix, pi-helix)
    E, B     →  E  (beta-strand, beta-bridge)
    T, S, ' '→  C  (turn, bend, loop/coil)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydssp
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1

log = logging.getLogger(__name__)

# Map 8-state DSSP codes to 3-state H/E/C
DSSP_TO_3STATE: dict[str, str] = {
    "H": "H",  # alpha-helix
    "G": "H",  # 3-10 helix
    "I": "H",  # pi-helix
    "E": "E",  # beta-strand
    "B": "E",  # isolated beta-bridge
    "T": "C",  # turn
    "S": "C",  # bend
    " ": "C",  # loop/coil
    "-": "C",  # no assignment (pydssp fallback)
}


def _extract_backbone(structure) -> tuple[np.ndarray, str]:
    """
    Extract backbone atom coordinates and amino acid sequence from a
    biopython Structure object.

    Returns
    -------
    coords : np.ndarray, shape (N_residues, 4, 3), dtype float32
        Backbone atoms in order: N, CA, C, O for each residue.
    sequence : str
        One-letter amino acid sequence.
    """
    atom_order = ("N", "CA", "C", "O")
    residue_coords: list[list] = []
    seq_chars: list[str] = []

    # Use only the first model and first chain to avoid duplicates
    model = next(iter(structure))
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            try:
                atoms = [res[a].get_vector().get_array() for a in atom_order]
            except KeyError:
                # Skip residues missing backbone atoms
                continue
            residue_coords.append(atoms)
            try:
                seq_chars.append(protein_letters_3to1[res.get_resname()])
            except Exception:
                seq_chars.append("X")
        break  # first chain only

    if not residue_coords:
        raise ValueError("No standard residues with complete backbone found.")

    coords = np.array(residue_coords, dtype=np.float32)  # (N, 4, 3)
    return coords, "".join(seq_chars)


def assign_dssp(pdb_path: Path) -> tuple[str, list[str]]:
    """
    Parse a PDB file and return per-residue sequence and 3-state SS labels.

    Parameters
    ----------
    pdb_path : Path
        Path to a PDB-format .ent file.

    Returns
    -------
    sequence : str
        One-letter amino acid sequence.
    ss_labels : list[str]
        Per-residue secondary structure labels: 'H', 'E', or 'C'.

    Raises
    ------
    ValueError
        If the file cannot be parsed or contains no standard residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    coords, sequence = _extract_backbone(structure)

    # pydssp.assign expects float32 (N, 4, 3) array: N, CA, C, O per residue
    raw_codes = pydssp.assign(coords)  # returns list/array of single chars

    ss_labels: list[str] = []
    for code in raw_codes:
        if isinstance(code, (int, np.integer)):
            # Some pydssp versions return integer codes; convert via index
            int_to_char = {
                0: " ",
                1: "H",
                2: "B",
                3: "E",
                4: "G",
                5: "I",
                6: "T",
                7: "S",
            }
            char = int_to_char.get(int(code), " ")
        else:
            char = str(code)
        ss_labels.append(DSSP_TO_3STATE.get(char, "C"))

    if len(ss_labels) != len(sequence):
        log.warning(
            "%s: DSSP returned %d labels for %d residues; truncating.",
            pdb_path.name,
            len(ss_labels),
            len(sequence),
        )
        n = min(len(ss_labels), len(sequence))
        ss_labels = ss_labels[:n]
        sequence = sequence[:n]

    return sequence, ss_labels
