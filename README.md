# Projet 2 — Controlled synthetic medical images (DermaMNIST)

Generate synthetic dermoscopy images conditioned on a class, compare a **CVAE** with a **conditional GAN**, and test whether the synthetic images help a classifier on the **rare class** (dermatofibroma), without copying the training images.

The full plan, with a mind map and progress checkboxes, is in [`impl.md`](impl.md).

## Status

| Step | Content | Status |
|---|---|---|
| S0 | Setup, repository structure | done |
| S1 | Data: DermaMNIST 64 × 64, class statistics | done |
| S2 | Baseline classifier (real data only) | code ready, full run on GPU pending |
| S3–S7 | Generative models, evaluation, demo, report | to do |

## Installation

**Locally (Windows, existing `.venv`):**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r mini_project\requirements.txt
```

**Google Colab:** open a notebook and run its first cell. It clones this repository and installs `requirements.txt` automatically. Choose a GPU runtime for training.

## Run

```bash
python data/download.py          # DermaMNIST 64x64 into data/ (~100 MB, not committed)
```
Then run the notebooks in order:

| Notebook | Step | Outputs in `results/` |
|---|---|---|
| `notebooks/01_data.ipynb` | S1 data exploration | class distribution, example images |
| `notebooks/02_baseline.ipynb` | S2 baseline classifier | precision / recall / F1 tables, confusion matrices |

Hyper-parameters are in `configs/`.

## Repository layout

```
configs/     experiment parameters (YAML)
data/        download script + data card (no data committed)
notebooks/   numbered notebooks
src/         reusable code (data, models, training, plots)
results/     figures and tables
demo/        demo app + video (S6)
report/      final report (S7)
docs/        licences, model card, ethics (S7)
```

## Data

DermaMNIST (MedMNIST v3, from HAM10000), licence **CC BY-NC 4.0**. See [`data/DATA_CARD.md`](data/DATA_CARD.md).

## Main results

*Filled in after the full runs.*
