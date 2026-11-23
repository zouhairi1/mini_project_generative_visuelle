# Team guide: what we did, why, and how to run it

This guide is for every team member who clones the repository. Read it once from top to bottom: it explains **what each file is for**, **why we made each choice**, and **how to run the code step by step** on your own PC.

---

## 1. The project in 30 seconds

**Goal (Projet 2):** generate synthetic dermoscopy images of a chosen class with two generative models (a **CVAE** and a **conditional GAN**), then check whether these synthetic images help a classifier on the **rare class**, without copying the training images.

**Dataset:** DermaMNIST (skin lesion images, 7 classes, 64 × 64). The rare class is **dermatofibroma (df)**: only 80 training images, versus 4 693 for the most common class.

**Where we are:**

| Step | What | Status |
|---|---|---|
| S0 | Repository and environment setup | done |
| S1 | Data exploration (`01_data.ipynb`) | done |
| S2 | Baseline classifier on real data only (`02_baseline.ipynb`) | done |
| S3 | Generative models: CVAE (`03a_cvae.ipynb`) and conditional GAN (`03b_cgan.ipynb`) | done |
| S4 | Synthetic sets (`04_generate.ipynb`) | done |
| S5–S7 | Evaluation, demo, report | to do |

The full plan, with a mind map and checkboxes, is in [`impl.md`](../impl.md).

---

## 2. What you need to install once

| Tool | Why |
|---|---|
| **Git** | to clone the repository and push your work |
| **Python 3.11** (3.10–3.12 also work) | to run the code |
| **VS Code** with the **Python** and **Jupyter** extensions | to open and run the notebooks |
| Optional: the **Colab extension for VS Code**, or just a browser with Google Colab | to train on a free GPU |
| A **GitHub account** added as collaborator on the repository | to push your commits (ask the repo owner) |

---

## 3. Step by step: set up the project on your PC

### Step 1 — Clone the repository
```powershell
git clone https://github.com/zouhairi1/mini_project_generative_visuelle.git
cd mini_project_generative_visuelle
```
**Why:** you get all the code, notebooks and results. The dataset and the trained models are **not** in Git (too large), you download or regenerate them in the next steps.

### Step 2 — Create a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux
```
**Why:** the project's libraries are installed in this folder only, so they do not conflict with other projects. `.venv/` is ignored by Git.

> If PowerShell refuses to run `Activate.ps1`, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

When the environment is active, your terminal line starts with `(.venv)`.

### Step 3 — Install the libraries
```powershell
pip install -r requirements.txt
```
**Why:** `requirements.txt` lists everything the code needs (PyTorch, MedMNIST, scikit-learn, matplotlib…). It takes a few minutes; `gradio` is only needed later for the demo.

### Step 4 — Download the dataset
```powershell
python data/download.py
```
**Why:** downloads DermaMNIST 64 × 64 (about 100 MB) into `data/dermamnist_64.npz`. You should see 7 007 / 1 003 / 2 005 images for train / val / test. This file is ignored by Git.

### Step 5 — Open the project in VS Code
1. **File → Open Folder** → choose `mini_project_generative_visuelle`.
2. Open a notebook in `notebooks/`.
3. Top right, **Select Kernel → Python Environments → `.venv`**.

---

## 4. Step by step: run the notebooks

Run them **in order**. Each notebook starts with a **setup cell** that finds the project folder (so `import src` works) and, on Colab, clones the repository and installs the libraries automatically.

### Notebook 1 — `01_data.ipynb` (S1): run it on your PC
- **Where:** your PC is enough (about 30 seconds, no training).
- **How:** select the `.venv` kernel → **Run All**.
- **What it does:** loads the data, counts images per class, shows examples and the mean image of each class.
- **Outputs:** `results/s1_*.csv` and `results/s1_*.png`.

### Notebook 2 — `02_baseline.ipynb` (S2): run it on a GPU
Training 9 models takes **about 6 hours on a laptop CPU** but **about 15 minutes on a Colab GPU**. Choose one of the options below.

**Option A — Quick test on your PC (to check that everything works)**
- In the third cell, set `QUICK = True` → **Run All**.
- It trains 1 model for 10 epochs (about 20 minutes on CPU). The numbers are only a test; **do not commit these results**.

**Option B — Full run from VS Code with a Colab GPU**
1. Select Kernel → **Colab → New Colab Server → GPU → T4**.
2. Keep `QUICK = False` → **Run All**.
3. Check that the cell after the setup prints `device: cuda`. If it says `cpu`, you did not get a GPU.
4. The code runs on Google's machine, so the result files are created **there**, not on your PC. The last cell prints them as text: **save the notebook (Ctrl+S)**, then in your local terminal (with `.venv` active, from the repository root):
   ```powershell
   python -m src.extract_results notebooks\02_baseline.ipynb
   ```
   This recreates the files in `results/` and `checkpoints/` and shortens the notebook again.

**Option C — Full run in the browser (Google Colab)**
1. Open `https://colab.research.google.com/github/zouhairi1/mini_project_generative_visuelle/blob/main/notebooks/02_baseline.ipynb`
2. **Runtime → Change runtime type → T4 GPU** → **Runtime → Run all**.
3. The second-to-last cell downloads `s2_results.zip`. Unzip it into your local repository folder.
4. The last cell (made for VS Code) prints a very long text in the browser: you can ignore it.

> **Important:** on Colab, the code is cloned from GitHub. It runs **what is pushed**, not your unsaved local changes. Push your code changes before running on Colab.

You do not have to rerun S2 unless you change the classifier: its results are already in `results/`. The only file you may need is `checkpoints/baseline_classifier.pt` (used in S5); ask the team for it (shared on Google Drive) or regenerate it with Option B or C.

### Notebooks 3a and 3b — `03a_cvae.ipynb` and `03b_cgan.ipynb` (S3): run them on a GPU
- **Where:** Colab GPU from VS Code (Option B). Run `03a` first, then `03b`.
- **No git needed:** the first cell of these notebooks contains a copy of `src/` and `configs/` from your laptop (`CODE_ZIP = "..."`), and writes it onto the Colab machine. **After changing any file in `src/` or `configs/`, update that copy** before running again:
  ```powershell
  python -m src.sync_code notebooks\03a_cvae.ipynb notebooks\03b_cgan.ipynb
  ```
  If the notebook is open in VS Code, reload it from disk afterwards (close and reopen it).
- **What they do:**
  - `03a` trains three conditional VAEs (β = 0.5, 1 and 2, 15 epochs each);
  - `03b` trains the conditional GAN (100 epochs, about 20 minutes; see section 6 for why not 15);
  - each one shows real vs synthetic images, the evolution during training, the diversity per class, a "class swap" (same z, different class) and latent interpolations.
- **Quick mode:** `QUICK = True` trains 10 epochs (and only β = 1 for the CVAE).
- **Getting the results back (VS Code):** the last cell prints a long text. Save the notebook (Ctrl+S), then run locally:
  ```powershell
  python -m src.extract_results notebooks\03a_cvae.ipynb
  python -m src.extract_results notebooks\03b_cgan.ipynb
  ```
- **Outputs:** `results/s3_cvae_*`, `results/s3_cgan_*`, and the models `checkpoints/cvae.pt` and `checkpoints/cgan_G.pt` (float16, used in S4; not in Git).
- If the GAN images are still rough, raise `epochs` in `configs/cgan.yaml` again, then run `src.sync_code` before running the notebook.

### Notebook 4 — `04_generate.ipynb` (S4): run it on your PC
- **Where:** your PC, with the **`.venv` kernel** (not Colab): it only uses the trained generators, about 1 minute on CPU.
- **Before:** `checkpoints/cvae.pt` and `checkpoints/cgan_G.pt` must exist (extracted from the S3 notebooks, or downloaded from the team's Google Drive).
- **How:** select the `.venv` kernel → **Run All**.
- **What it does:** generates 800 images per class with each model (fixed seed 2026), saves each set in its own file, checks that regenerating gives the same images, checks that val and test contain no synthetic image, and shows real vs CVAE vs cGAN plus watermarked examples.
- **Outputs:** `data/synthetic/cvae_n800_seed2026.npz` and `data/synthetic/cgan_n800_seed2026.npz` (**not in Git**, about 1 minute to regenerate), `results/s4_*`.

---

## 5. Role of each file

### Files written by hand

| File | Role | Why we need it |
|---|---|---|
| `README.md` | Front page: presentation, installation, main results | Required by the brief; the first thing the professor reads |
| `impl.md` | Full plan: mind map, steps S0–S7 with checkboxes, course references | Keeps the team aligned; tick a box when a task is finished |
| `requirements.txt` | List of Python libraries | Anyone can recreate the same environment in one command |
| `.gitignore` | Files Git must ignore (data, models, `.venv`) | Large files must not go on GitHub (brief rule) |
| `configs/classifier.yaml` | S2 settings: epochs, learning rate, seeds, strategies | Change experiments without touching the code; every run is documented |
| `data/download.py` | Downloads DermaMNIST | We do not commit the data, only how to get it |
| `data/DATA_CARD.md` | Data sheet: source, licence, class counts, biases | Required by the brief (`data/` = script + data sheet) |
| `src/utils.py` | Paths, random seed, device (GPU/CPU), config loading | Shared helpers used by every notebook |
| `src/data.py` | Loads the dataset, normalises pixels to [-1, 1], creates the data loaders (with random flips and optional oversampling) | Same data pipeline everywhere, written once |
| `src/models.py` | The CNN classifier (`SmallCNN`) | Reusable model; its `features()` method is used later to compare real and synthetic images |
| `src/train_clf.py` | Training loop, metrics (precision, recall, F1, confusion matrix) | Same training and evaluation for S2 and the S5 experiments |
| `src/viz.py` | Plots: examples per class, confusion matrices | Same figures style everywhere |
| `src/extract_results.py` | Recovers result files from a notebook run on Colab via VS Code | With the VS Code Colab kernel, files stay on Google's machine |
| `configs/cvae.yaml`, `configs/cgan.yaml` | S3 settings: 15 epochs, learning rates, β values, label smoothing, sampler | Same reason as the classifier config: change experiments without touching the code |
| `src/train_gen.py` | S3 training loops (CVAE, conditional GAN), sample generation, diversity measure | Both generative models trained and evaluated the same way |
| `notebooks/03a_cvae.ipynb` | S3a: conditional VAE + β experiment | First generative model (course: AE/VAE) |
| `notebooks/03b_cgan.ipynb` | S3b: conditional GAN (course version) | Second generative model (course: GAN), compared with the CVAE |
| `src/sync_code.py` | Copies `src/` and `configs/` into the first cell of the S3 notebooks | Lets the VS Code Colab kernel use the laptop code without git |
| `configs/generate.yaml` | S4 settings: models, 800 images per class, seed 2026 | Fixed, documented generation |
| `src/generate.py` | Loads the S3 generators, generates reproducible sets, saves / loads them, leak check, `SYNTHETIC` watermark | Synthetic data is created and handled in one place, with the brief's rules built in |
| `notebooks/04_generate.ipynb` | S4: builds and checks the synthetic sets | Fixed datasets for the S5 experiments |
| `notebooks/01_data.ipynb` | S1: data exploration | Understand the data and find the rare class |
| `notebooks/02_baseline.ipynb` | S2: baseline classifier | Reference performance that synthetic images must beat |
| `docs/GUIDE.md` | This guide | Lets every member set up and run the project alone |

Empty folders for later steps: `demo/` (S6 app + video), `report/` (final report), `docs/` (licences, model card, ethics).

### Files generated by the code

| File | Created by | In Git? | Content |
|---|---|---|---|
| `data/dermamnist_64.npz` | `data/download.py` or any notebook | **no** (100 MB) | The dataset |
| `results/s1_class_distribution.csv/.png` | `01_data` | yes | Number of images per class and split |
| `results/s1_samples_per_class.png` | `01_data` | yes | 8 example images per class |
| `results/s1_mean_image_per_class.png` | `01_data` | yes | Average image of each class |
| `results/s2_summary.csv` | `02_baseline` | yes | Macro and df precision / recall / F1 per strategy (mean ± std) |
| `results/s2_per_class.csv` | `02_baseline` | yes | Precision / recall / F1 for every class |
| `results/s2_confusion_matrices.png` | `02_baseline` | yes | Which classes are confused with which |
| `results/s2_training_curves.png` | `02_baseline` | yes | Loss and validation scores per epoch |
| `results/s2_baseline_runs.json` | `02_baseline` | yes | All raw numbers of the 9 runs |
| `checkpoints/clf_<strategy>_seed<n>.pt` | `02_baseline` | **no** | The 9 trained classifiers |
| `checkpoints/baseline_classifier.pt` | `02_baseline` | **no** (share on Drive) | Reference classifier, used in S5 to evaluate the generated images |
| `results/s3_cvae_*` | `03a_cvae` | yes | CVAE losses, samples, evolution, reconstructions, β table, diversity, class swap, interpolation |
| `results/s3_cgan_*` | `03b_cgan` | yes | cGAN losses and D outputs, samples, evolution, diversity, class swap, interpolation |
| `checkpoints/cvae.pt`, `checkpoints/cgan_G.pt` | `03a`, `03b` | **no** (share on Drive) | Main CVAE (β = 1) and cGAN generator, float16, used in S4 |
| `data/synthetic/<model>_n800_seed2026.npz` | `04_generate` | **no** (regenerate in 1 min) | The synthetic sets: images, labels, flag `synthetic=True` |
| `results/s4_*` | `04_generate` | yes | Set sizes, real vs CVAE vs cGAN grid, watermarked examples |

---

## 6. Why we made these choices

| Choice | Why |
|---|---|
| **DermaMNIST** (dermoscopy) instead of PneumoniaMNIST | 7 classes with a strong imbalance, so there is a real rare class to help; PneumoniaMNIST has only 2 classes |
| **64 × 64** images | Recommended by the brief; trains fast and fits easily in 12 GB of GPU memory |
| **Official train / val / test split** | Required by the brief; results are comparable with other work |
| **Rare class = dermatofibroma (df)** | The smallest class (80 train images, 1.1 % of the data) |
| Pixels in **[-1, 1]** | Matches the `tanh` output of the generators we will build (GAN course) |
| Random **horizontal / vertical flips** | A lesion seen in a mirror is still the same lesion: free extra variety |
| **Three strategies** in S2: `none`, `weighted`, `oversample` | `none` is the honest baseline; `weighted` and `oversample` are the classic fixes for imbalance. Synthetic images must do better than these simple fixes to be worth it |
| **3 seeds**, results as mean ± std | Only 23 df test images: one image changes df recall by about 4 points, so a single run is not reliable |
| Best epoch chosen on **validation macro-F1** | Macro-F1 gives the same weight to every class, so the rare class counts; the test set is used only once at the end |
| Metrics: **precision, recall, F1, confusion matrix** | Accuracy would be misleading: predicting "nv" every time already gives 67 % accuracy |
| Keep the best `none` model as **reference classifier** | Its features are specific to skin images; we use it in S5 to measure the quality of the generated images (the GAN course notes that ImageNet features suit medical images poorly) |
| Data and models **not in Git** | Brief rule: no large files in the repository |
| **Same generator** for the CVAE decoder and the cGAN | Fair comparison: only the training objective differs (ELBO vs adversarial game) |
| cGAN = **course version** (BCE, non-saturating loss, label smoothing 0.9, Adam 2e-4 / β1 0.5, DCGAN layers) | Every choice can be justified from the GAN course during the presentation |
| **β = 0.5 / 1 / 2** for the CVAE | β-VAE slide of the course: shows the trade-off between sharp reconstructions and a smooth latent space |
| **sqrt-balanced sampler** for S3 | Rare classes are seen more often, but less than with full oversampling, which would push the models to memorize the 80 df images |
| **CVAE: 15 epochs** | Enough: the reconstruction loss is almost flat at the end, and a VAE learns quickly because every image gets a direct reconstruction error |
| **cGAN: first 15 epochs, then 100** | See below |
| **log σ² clamped to [-10, 10]** in the CVAE encoder | In the first run the KL reached ~7 billion at epoch 1 because of the random initial weights (it recovered at epoch 2). The clamp prevents this explosion, which could otherwise turn into NaN; trained values stay far inside the range |

### Why the cGAN went from 15 to 100 epochs

We first trained **both models for 15 epochs**, the same budget for a fair first comparison and to keep runs short.

- **CVAE after 15 epochs:** good. The losses had flattened, the β experiment followed the theory, and the images were blurry as the course predicts (the decoder outputs the average of possible images).
- **cGAN after 15 epochs:** the training itself was healthy (at the end D(x) ≈ 0.48 and D(G(z)) ≈ 0.42, close to the ideal 0.5; no mode collapse), **but the images were not usable**:
  - a regular **grid pattern** (checkerboard artifact of the transposed convolutions, typical of an under-trained GAN);
  - a **purple / blue tint** that does not exist in the real images;
  - almost **no class control**: with the same z, all classes looked the same.
- **Why:** 15 epochs is only about **760 training steps**. A VAE gets a direct, pixel-by-pixel error signal on every image, so it learns quickly. A GAN only learns through the discriminator's judgement, in a two-player game, so it needs many more steps to learn colours, textures and the effect of the class. DCGAN-type models usually need thousands of steps.
- **Decision:** **100 epochs** (about **5 000 steps**). On a T4 GPU one epoch takes ~12 s, so the run takes about **20 minutes**. The CVAE stays at 15 epochs.
- **Result of the 100-epoch run:** the class now controls the image (mel dark brown/black, bkl brown, vasc purple/magenta, nv pink-red, df pale pink-brown), the purple tint is gone and only a faint grid pattern remains. Diversity ratio 0.90–1.08: no mode collapse. Training stayed balanced (D(x) 0.46, D(G(z)) 0.44).
- **Could 50 epochs be enough?** The evolution figure (`results/s3_cgan_evolution.png`) shows epochs 1, 51 and 100: epoch 100 is clearly sharper and more class-specific than epoch 51, so we keep 100.
- **Note on the diversity ratio:** after 15 epochs the cGAN had high pixel diversity (0.74–1.06), but this was misleading: noise and the grid pattern count as "diversity" when images are compared pixel by pixel. Always look at the images, not only at the numbers (GAN course, slide 18).

**S2 result in one sentence:** trained on real data only, the classifier finds only about **12 %** of df images (df F1 0.205); oversampling raises df F1 to **0.464**. The synthetic images of S3–S5 will have to beat oversampling.

---

## 7. Team rules (Git)

1. **Before you start working:** `git pull` to get the latest version.
2. **Do not commit** the data, `.pt` files, `.venv/`, or results from a `QUICK = True` run.
3. **Commit often, with a clear message**, for example:
   - `feat(cvae): add encoder and decoder`
   - `results: S3 CVAE samples`
   - `docs: update data card`
4. **One person per notebook at a time:** Git cannot merge two versions of the same notebook easily. Agree in the group chat (or a GitHub issue) who works on which notebook.
5. **Everyone commits from their own account**, so each member's contribution is visible (the brief requires it).
6. When you finish a task, **tick its box in `impl.md`** in the same commit.

---

## 8. Common problems

| Problem | Solution |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `ModuleNotFoundError: No module named 'src'` | Run the notebook's first (setup) cell; open the repository root folder in VS Code, not a sub-folder |
| `ModuleNotFoundError: medmnist` (or another library) | The `.venv` kernel is not selected, or `pip install -r requirements.txt` was not run |
| Colab prints `device: cpu` | You did not get a GPU: create a new Colab server and choose **GPU** |
| `s2_results.zip not found` | The Colab session ended and its files were deleted: **Run All** again |
| `No ZIP_BASE64 output found` when extracting | You forgot to save the notebook (Ctrl+S) after the last cell ran |
| Colab runs old code | Colab clones from GitHub: push your changes first |
