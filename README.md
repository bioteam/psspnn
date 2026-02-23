# psspnn

Protein secondary structure prediction using the neural network described in:

> Holley, L.H. & Karplus, M. (1989). Protein secondary structure prediction with a neural network. *Proc. Natl. Acad. Sci. USA* 86, 152–156.

## Architecture

The network is a feed-forward neural network with sigmoid activations:

```
Input (17 × 21 = 357 units)
   ↓ sliding window of 17 residues, one-hot encoded
   ↓ 21 bits per position: 20 amino acids + 1 null (terminal padding)
Hidden (2 units, default)
   ↓
Output (2 units)
   (1,0) = helix   (0,1) = sheet   (0,0) = coil
```

Training uses full-batch backpropagation (Rumelhart et al. 1986) minimising the sum of squared errors. Training stops when the fractional change in loss per cycle drops below 2×10⁻⁴.

Post-processing applies minimum-length rules from Kabsch & Sander (1983): helix runs < 4 residues and strand runs < 2 residues are reclassified as coil.

The 62 benchmark proteins are those used in Kabsch & Sander (1983) FEBS Lett 155:179, split into 48 training and 14 test proteins as in the original paper.

## Requirements

- Python ≥ 3.10
- numpy, biopython, pydssp, click

## Installation

```bash
git clone https://github.com/bioteam/psspnn.git
cd psspnn
poetry install
```

Or with pip:

```bash
pip install .
```

## Usage

### 1. Download the benchmark PDB files

```bash
psspnn download --split all
```

Files are cached to `~/.psspnn/pdb_cache/`. Re-running is safe; already-cached files are skipped.

### 2. Train

```bash
psspnn train --verbose
```

Trains on the 48 proteins from the training set. Progress is printed every 10 cycles. The model is saved to `~/.psspnn/model.npz`.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--window` | 17 | Input window size |
| `--hidden` | 2 | Hidden layer size (0 = no hidden layer) |
| `--lr` | 0.1 | Learning rate |
| `--max-cycles` | 2000 | Maximum training cycles |
| `--tol` | 2e-4 | Fractional-change stopping threshold |
| `--seed` | — | Random seed for reproducibility |
| `--model-out` | `~/.psspnn/model.npz` | Output path |

### 3. Evaluate

```bash
psspnn evaluate --split test
```

Prints per-protein percent-correct and aggregate quality indices (percent correct, Matthews correlation coefficients, PC(S)) against the expected values from Table 1 of the paper (~63% on the test set).

### 4. Predict

```bash
psspnn predict ACDEFGHIKLMNPQRSTVWY
```

Outputs predictions as a string of H (helix), E (sheet), C (coil):

```
Sequence:   ACDEFGHIKLMNPQRSTVWY
Prediction: CCCCCCCCCCCCCCCCCCCC
```

Use `--format table` for per-residue output with raw network activations.

## Expected results

| Metric | Training set | Test set (paper) |
|--------|-------------|----------|
| Percent correct | 68.5% | 63.2% |
| C_helix | 0.49 | 0.41 |
| C_sheet | 0.47 | 0.32 |
| C_coil | 0.43 | 0.36 |
| PC(helix) | 65.3% | 59.2% |
| PC(sheet) | 63.4% | 53.3% |
| PC(coil) | 71.1% | 66.9% |

## Package structure

```
src/psspnn/
├── data/
│   ├── protein_ids.py   Ordered list of 62 benchmark PDB IDs
│   ├── download.py      PDB file retrieval via biopython
│   ├── dssp.py          DSSP assignment via pydssp → H/E/C labels
│   └── encoding.py      One-hot window encoding and dataset assembly
├── model/
│   ├── network.py       HolleyKarplusNet (forward pass, save/load)
│   ├── training.py      Full-batch backprop with SSE loss
│   └── predict.py       Threshold + contiguity post-processing
├── evaluation/
│   └── metrics.py       Matthews CC, percent correct, PC(S)
└── cli.py               Command-line interface
```

## Running tests

```bash
poetry run pytest
```

## Notes

- The learning rate (0.1) is not stated in the paper. The value from Rumelhart et al. (1986) is used as the default. Adjust with `--lr` if convergence is slow.
- The 48 training protein IDs are sourced from Kabsch & Sander (1983) FEBS Lett 155:179 table 1. Some 1983-era PDB entries have been superseded; failed downloads are logged as warnings.
- The DSSP 3-state mapping used is: {H, G, I} → H, {E, B} → E, {T, S, ' '} → C.

## Reference

Holley, L.H. & Karplus, M. (1989). Protein secondary structure prediction with a neural network. *Proc. Natl. Acad. Sci. USA* **86**, 152–156. [doi:10.1073/pnas.86.1.152](https://doi.org/10.1073/pnas.86.1.152)
