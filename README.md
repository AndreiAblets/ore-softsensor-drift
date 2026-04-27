# Ore Soft-Sensor Drift

[![DOI](https://zenodo.org/badge/1222806124.svg)](https://doi.org/10.5281/zenodo.19828095)

This repository contains a compact reproduction pipeline for calibration-drift robustness experiments on a public ore grinding and grading dataset.

The code evaluates source-trained soft sensors under small multiplicative sensor drift. It identifies drift-sensitive channels and tests three remediation strategies: feature removal, batch moment correction, and deployment-time CORAL alignment.

The feature-removal remedy selects the removed channel from the training split by the largest vulnerability index. No sensor index is hardcoded.

## Repository layout

```text
ore-softsensor-drift/
  data/                         Dataset instructions
  scripts/run_reproduction.py   Main entry point
  src/ore_softsensor_drift/     Reproduction code
  requirements.txt              Python dependencies
  CITATION.cff                  Repository citation metadata
  LICENSE                       Code license
```

## Data

The data are not bundled with this repository. Download version 1 of the public Mendeley dataset:

```text
Dataset on ore grinding and grading process
DOI: 10.17632/hgsf7bwkrv.1
URL: https://doi.org/10.17632/hgsf7bwkrv.1
```

Place the four CSV files in:

```text
data/hgsf7bwkrv-1/
  X_src.csv
  Y_src.csv
  X_tar.csv
  Y_tar.csv
```

The Mendeley Data record lists this dataset under CC BY 4.0. This repository only contains code.

The public metadata do not define the ore type, mill type, class semantics, sampling rate, or source/target split criterion. The code uses the numeric labels and the released source/target files as provided. The loader removes the all-zero first row from each feature matrix and uses the remaining eight measured rows.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.10.11, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.8, and torch 2.11.0.

## Run

Full reproduction:

```bash
python scripts/run_reproduction.py --data-dir data/hgsf7bwkrv-1 --output-dir outputs --seeds 0,1,2,3,4
```

Quick smoke run:

```bash
python scripts/run_reproduction.py --data-dir data/hgsf7bwkrv-1 --output-dir outputs_quick --seeds 0 --fast
```

The script writes tables to `outputs/tables/` and figures to `outputs/figures/`.

## Main outputs

- `baseline_mean_std.csv`
- `drift_global_mean_std.csv`
- `drift_per_channel_005.csv`
- `vulnerability_index.csv`
- `high_vi_selection.csv`
- `remedies_mean_std.csv`
- `solutions_mean_std.csv`
- `coral_under_drift.csv`
- `conformal_solutions_aps.csv`
- `drift_detection.csv`
- `regime_ood_mean_std.csv`

## License

Code is released under the MIT License. The dataset remains governed by the dataset provider's license.
