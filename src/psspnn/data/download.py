"""
Download PDB files from the RCSB Protein Data Bank.

Uses biopython's PDBList for retrieval. Files are cached locally; if a file
already exists it is not re-downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

from Bio.PDB import PDBList

log = logging.getLogger(__name__)


def fetch_pdb(pdb_id: str, cache_dir: Path) -> Path | None:
    """
    Download a PDB file to cache_dir if not already present.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier (case-insensitive).
    cache_dir : Path
        Directory where PDB files are cached.

    Returns
    -------
    Path to the downloaded .ent file, or None if the download failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_id_lower = pdb_id.lower()
    dest = cache_dir / f"pdb{pdb_id_lower}.ent"
    if dest.exists():
        return dest

    pl = PDBList(verbose=False)
    try:
        result = pl.retrieve_pdb_file(
            pdb_id,
            file_type="pdb",
            pdir=str(cache_dir),
            overwrite=False,
        )
        if result:
            result_path = Path(result)
            if result_path.exists():
                # PDBList may write to a subdirectory; move to flat cache_dir
                if result_path != dest:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    result_path.rename(dest)
                return dest
    except Exception as exc:
        log.warning("Failed to download %s: %s", pdb_id, exc)

    log.warning("PDB entry %s could not be retrieved.", pdb_id)
    return None


def fetch_all(
    pdb_ids: list[str],
    cache_dir: Path,
) -> dict[str, Path | None]:
    """
    Fetch multiple PDB files, returning a mapping of pdb_id -> Path (or None).

    Parameters
    ----------
    pdb_ids : list[str]
    cache_dir : Path

    Returns
    -------
    dict mapping each PDB ID to its local Path, or None on failure.
    """
    results: dict[str, Path | None] = {}
    for pid in pdb_ids:
        results[pid] = fetch_pdb(pid, cache_dir)
    return results
