# psspnn

This code reproduces the protein secondary structure prediction neural network described in Holley, L.H. & Karplus, M. (1989). Protein secondary structure prediction with a neural network. *Proc. Natl. Acad. Sci. USA* **86**, 152–156. [doi:10.1073/pnas.86.1.152](https://doi.org/10.1073/pnas.86.1.152) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/2911565/)

## Architecture

The network is a feed-forward neural network with sigmoid activations:

```shell
Input (17 × 21 = 357 units)
   ↓ sliding window of 17 residues, one-hot encoded
   ↓ 21 bits per position: 20 amino acids + 1 null (terminal padding)
Hidden (2 units, default)
   ↓
Output (2 units)
   (0.9, 0.1) = helix   (0.1, 0.9) = sheet   (0.1, 0.1) = coil
```

Training uses full-batch backpropagation (Rumelhart et al. 1986 [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0)) minimising the sum of squared errors. Training stops when the fractional change in loss per cycle drops below 2×10⁻⁴.

The 62 benchmark proteins are those used in Kabsch & Sander (1983) FEBS Lett 155:179 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/6852232/)), split into 47 training and 14 test proteins as in the original paper.

Post-processing applies minimum-length rules from Kabsch & Sander (1983): helix runs < 4 residues and strand runs < 2 residues are reclassified as coil.

### Output

With 2 output units you can encode 3 classes using a binary scheme:

| Output 1 | Output 2 | Class |
| -------- | -------- | ----- |
| 0.9 | 0.1 | Helix (H) |
| 0.1 | 0.9 | Sheet (E) |
| 0.1 | 0.1 | Coil (C) |

Offset targets (0.1/0.9 instead of 0/1) prevent sigmoid saturation, which is standard practice for sigmoid-output networks (Rumelhart et al. 1986). Coil is the "default" — it's what you get when neither output unit is strongly activated, coil is essentially "neither helix nor sheet," so two binary decisions are sufficient to distinguish all three states. A 3-unit one-hot output layer (one neuron per class) would also work but would add extra parameters.

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

Trains on the 47 proteins from the training set. Progress is printed every cycle. The model is saved to `~/.psspnn/model.npz`.

Options:

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--window` | 17 | Input window size |
| `--hidden` | 2 | Hidden layer size (0 = no hidden layer) |
| `--lr` | 10.0 | Learning rate |
| `--max-cycles` | 2000 | Maximum training cycles |
| `--tol` | 2e-4 | Fractional-change stopping threshold |
| `--seed` | — | Random seed for reproducibility |
| `--model-out` | `~/.psspnn/model.npz` | Output path |

The training proteins have varying lengths (47 proteins totalling ~8,315 residues, average ~173 per protein). For each residue in a protein, a window of 17 residues centred on that position is extracted and one-hot encoded as a 17×21 = 357-element input vector. That window becomes one training sample, with the target being the DSSP label (H/E/C) of the central residue.

### 3. Evaluate

```bash
psspnn evaluate --split test
```

Prints per-protein percent-correct and aggregate quality indices (percent correct, Matthews correlation coefficients, PC(S)) against the expected values from Table 1 of the paper (~63% on the test set).

### 4. Predict

```bash
psspnn predict LFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILK
```

Outputs predictions as a string of H (helix), E (sheet), C (coil):

```shell
Sequence:   LFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILK
Prediction: CCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHCEEEHHHHHCCCC
```

Use `--format table` for per-residue output with raw network activations.

## Expected results

| Metric | Training set | Test set (paper) |
| -------- | ------------- | ---------- |
| Percent correct | 68.5% | 63.2% |
| C_helix | 0.49 | 0.41 |
| C_sheet | 0.47 | 0.32 |
| C_coil | 0.43 | 0.36 |
| PC(helix) | 65.3% | 59.2% |
| PC(sheet) | 63.4% | 53.3% |
| PC(coil) | 71.1% | 66.9% |

## Package structure

```shell
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

- The learning rate is not stated in the paper. Rumelhart et al. (1986) used 0.1 for per-pattern updates; since this implementation uses full-batch training with mean gradients, the default is 10.0. Adjust with `--lr` if convergence is slow.
- The training protein IDs are sourced from Kabsch & Sander (1983) FEBS Lett 155:179 table 1. Some 1983-era PDB entries have been superseded; failed downloads are logged as warnings.
- The DSSP (Define Secondary Structure of Proteins; Kabsch & Sander, 1983) 8-to-3 state mapping used is: {H, G, I} → H, {E, B} → E, {T, S, ' '} → C.
