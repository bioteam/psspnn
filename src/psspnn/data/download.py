"""
Download PDB files from the RCSB Protein Data Bank.

Downloads PDB-format files directly from files.rcsb.org. Files are cached
locally; if a file already exists it is not re-downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError

log = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


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
    Path to the downloaded .pdb file, or None if the download failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_id_upper = pdb_id.upper()
    dest = cache_dir / f"pdb{pdb_id.lower()}.pdb"
    if dest.exists():
        return dest

    url = RCSB_URL.format(pdb_id=pdb_id_upper)
    try:
        urlretrieve(url, dest)
        return dest
    except (HTTPError, OSError) as exc:
        log.warning("Failed to download %s: %s", pdb_id, exc)
        # Clean up partial file
        dest.unlink(missing_ok=True)

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
