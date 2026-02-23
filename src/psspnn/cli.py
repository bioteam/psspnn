"""
Command-line interface for psspnn.

Commands
--------
    psspnn download   Download PDB files for the Holley & Karplus proteins
    psspnn train      Train the network on the 48 training proteins
    psspnn predict    Predict secondary structure for a given sequence
    psspnn evaluate   Score a trained model on the training or test split
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

DEFAULT_CACHE = Path.home() / ".psspnn" / "pdb_cache"
DEFAULT_MODEL = Path.home() / ".psspnn" / "model.npz"

logging.basicConfig(
    format="%(levelname)s %(name)s: %(message)s",
    level=logging.WARNING,
)


@click.group()
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
def main(debug: bool) -> None:
    """psspnn: Holley & Karplus 1989 secondary structure neural network."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE,
    show_default=True,
    help="Directory where PDB files are cached.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "all"]),
    default="all",
    show_default=True,
)
def download(cache_dir: Path, split: str) -> None:
    """Download PDB files for the Holley & Karplus benchmark proteins."""
    from .data.protein_ids import get_split
    from .data.download import fetch_all

    ids = get_split(split)
    click.echo(f"Downloading {len(ids)} PDB files to {cache_dir} …")
    results = fetch_all(ids, cache_dir)
    failed = [pid for pid, path in results.items() if path is None]
    n_ok = len(ids) - len(failed)
    click.echo(f"Done: {n_ok}/{len(ids)} downloaded.")
    if failed:
        click.echo(f"Failed ({len(failed)}): {', '.join(failed)}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE,
    show_default=True,
)
@click.option(
    "--model-out",
    type=click.Path(path_type=Path),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Output path for the model weights (.npz).",
)
@click.option("--window", default=17, show_default=True, help="Input window size.")
@click.option("--hidden", default=2, show_default=True, help="Hidden layer size (0 = no hidden layer).")
@click.option("--lr", default=0.1, show_default=True, help="Learning rate.")
@click.option("--max-cycles", default=2000, show_default=True)
@click.option("--tol", default=2e-4, show_default=True, help="Fractional-change stopping threshold.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--verbose/--no-verbose", default=True, show_default=True)
def train(
    cache_dir: Path,
    model_out: Path,
    window: int,
    hidden: int,
    lr: float,
    max_cycles: int,
    tol: float,
    seed: int | None,
    verbose: bool,
) -> None:
    """Train the network on the 48 training proteins."""
    from .data.protein_ids import get_split
    from .data.download import fetch_pdb
    from .data.dssp import assign_dssp
    from .data.encoding import build_dataset
    from .model.network import HolleyKarplusNet
    from .model.training import train as _train

    pdb_ids = get_split("train")
    proteins: list[tuple[str, list[str]]] = []

    click.echo("Loading training proteins …")
    for pid in pdb_ids:
        pdb_path = fetch_pdb(pid, cache_dir)
        if pdb_path is None:
            click.echo(f"  SKIP {pid} (not found)", err=True)
            continue
        try:
            seq, ss = assign_dssp(pdb_path)
            proteins.append((seq, ss))
            if verbose:
                click.echo(f"  {pid}: {len(seq)} residues")
        except Exception as exc:
            click.echo(f"  SKIP {pid}: {exc}", err=True)

    if not proteins:
        click.echo("No proteins loaded. Run 'psspnn download' first.", err=True)
        sys.exit(1)

    click.echo(f"Building dataset from {len(proteins)} proteins …")
    X_train, y_train = build_dataset(proteins, window_size=window)
    click.echo(f"  X shape: {X_train.shape}  y shape: {y_train.shape}")

    # Report class composition
    n = len(y_train)
    n_h = int((y_train[:, 0] == 1).sum())
    n_e = int((y_train[:, 1] == 1).sum())
    n_c = n - n_h - n_e
    click.echo(
        f"  Composition: {n_h/n:.0%} H  {n_e/n:.0%} E  {n_c/n:.0%} C"
        f"  (paper: 26% H  20% E  54% C)"
    )

    click.echo("Training …")
    net = HolleyKarplusNet(window_size=window, hidden_units=hidden, seed=seed)
    result = _train(
        net, X_train, y_train,
        learning_rate=lr, max_cycles=max_cycles, tol=tol, verbose=verbose,
    )

    status = "converged" if result.converged else f"reached max_cycles={max_cycles}"
    click.echo(
        f"Training complete: {result.n_cycles} cycles, "
        f"E={result.final_error:.4f} ({status})"
    )

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    net.save(model_out)
    click.echo(f"Model saved to {model_out}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@main.command()
@click.argument("sequence")
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_MODEL,
    show_default=True,
)
@click.option("--threshold", default=0.37, show_default=True)
@click.option("--window", default=17, show_default=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["compact", "table"]),
    default="compact",
    show_default=True,
    help="Output format.",
)
def predict(
    sequence: str,
    model_path: Path,
    threshold: float,
    window: int,
    output_format: str,
) -> None:
    """
    Predict secondary structure for a one-letter amino acid SEQUENCE.

    Outputs per-residue predictions as H (helix), E (sheet), or C (coil).
    """
    from .model.network import HolleyKarplusNet
    from .model.predict import predict_sequence

    net = HolleyKarplusNet.load(Path(model_path))
    raw_out, final_preds = predict_sequence(
        net, sequence.upper(), window_size=window, threshold=threshold
    )

    if output_format == "compact":
        click.echo(f"Sequence:   {sequence.upper()}")
        click.echo(f"Prediction: {''.join(final_preds)}")
    else:
        click.echo(f"{'Pos':>4}  {'AA':>2}  {'Pred':>4}  {'Helix':>7}  {'Sheet':>7}")
        for i, (aa, pred, (h, s)) in enumerate(
            zip(sequence.upper(), final_preds, raw_out), start=1
        ):
            click.echo(f"{i:4d}  {aa:>2}  {pred:>4}  {h:7.4f}  {s:7.4f}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE,
    show_default=True,
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_MODEL,
    show_default=True,
)
@click.option("--threshold", default=0.37, show_default=True)
@click.option(
    "--split",
    type=click.Choice(["train", "test"]),
    default="test",
    show_default=True,
)
def evaluate(
    cache_dir: Path,
    model_path: Path,
    threshold: float,
    split: str,
) -> None:
    """Evaluate a trained model and print Table 1 / Table 2 style metrics."""
    from .data.protein_ids import get_split
    from .data.download import fetch_pdb
    from .data.dssp import assign_dssp
    from .model.network import HolleyKarplusNet
    from .model.predict import predict_sequence
    from .evaluation.metrics import full_evaluation, percent_correct

    net = HolleyKarplusNet.load(Path(model_path))
    pdb_ids = get_split(split)

    all_pred: list[str] = []
    all_actual: list[str] = []

    click.echo(f"\n{'ID':>6}  {'N':>5}  {'%Correct':>8}")
    click.echo("-" * 25)

    for pid in pdb_ids:
        pdb_path = fetch_pdb(pid, cache_dir)
        if pdb_path is None:
            click.echo(f"  SKIP {pid} (not found)", err=True)
            continue
        try:
            seq, ss_actual = assign_dssp(pdb_path)
        except Exception as exc:
            click.echo(f"  SKIP {pid}: {exc}", err=True)
            continue

        _, preds = predict_sequence(net, seq, threshold=threshold)
        pct = percent_correct(preds, ss_actual)
        click.echo(f"{pid:>6}  {len(seq):>5}  {pct:8.1f}")
        all_pred.extend(preds)
        all_actual.extend(ss_actual)

    if not all_pred:
        click.echo("No proteins evaluated.", err=True)
        sys.exit(1)

    metrics = full_evaluation(all_pred, all_actual)
    click.echo("-" * 25)
    click.echo(f"\nAggregate metrics ({split} set):")
    click.echo(f"  Percent correct:  {metrics['percent_correct']:.1f}%"
               f"  (paper: 63.2% test / 68.5% train)")
    click.echo(f"  C_helix:          {metrics['C_helix']:.2f}"
               f"  (paper: 0.41 test)")
    click.echo(f"  C_sheet:          {metrics['C_sheet']:.2f}"
               f"  (paper: 0.32 test)")
    click.echo(f"  C_coil:           {metrics['C_coil']:.2f}"
               f"  (paper: 0.36 test)")
    click.echo(f"  PC(helix):        {metrics['PC_helix']:.1f}%"
               f"  (paper: 59.2% test)")
    click.echo(f"  PC(sheet):        {metrics['PC_sheet']:.1f}%"
               f"  (paper: 53.3% test)")
    click.echo(f"  PC(coil):         {metrics['PC_coil']:.1f}%"
               f"  (paper: 66.9% test)")
