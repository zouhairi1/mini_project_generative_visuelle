# Data card: DermaMNIST (MedMNIST+, 64 × 64)

## Source
- **Dataset:** DermaMNIST, part of MedMNIST v3 (Yang et al., *MedMNIST v2*, Scientific Data, 2023).
- **Original images:** HAM10000 (Tschandl et al., Scientific Data, 2018): 10 015 dermatoscopic images of common pigmented skin lesions, collected from several sources.
- **Licence:** CC BY-NC 4.0 (non-commercial use, attribution required).
- **Download:** `python data/download.py` (file `dermamnist_64.npz`, about 100 MB, **not committed**).

## Format
- RGB images, **64 × 64**, `uint8` (resized from 3 × 600 × 450 by MedMNIST).
- One label per image, 7 classes (multi-class classification).
- Official split (ratio 7 : 1 : 2), used **as is**.

## Class distribution

| id | short | class | train | val | test |
|---|---|---|---:|---:|---:|
| 0 | akiec | actinic keratoses / intraepithelial carcinoma | 228 | 33 | 66 |
| 1 | bcc | basal cell carcinoma | 359 | 52 | 103 |
| 2 | bkl | benign keratosis-like lesions | 769 | 110 | 220 |
| **3** | **df** | **dermatofibroma (rare class)** | **80** | **12** | **23** |
| 4 | mel | melanoma | 779 | 111 | 223 |
| 5 | nv | melanocytic nevi | 4 693 | 671 | 1 341 |
| 6 | vasc | vascular lesions | 99 | 14 | 29 |
| | | **total** | **7 007** | **1 003** | **2 005** |

- **Imbalance:** nv is 67% of the training set; df is 1.1%, about **59× fewer** images than nv.
- **Rare class for this project:** dermatofibroma (class 3). Vascular lesions (class 6) is the second rarest.

## Use in this project
- Training generative models (CVAE, cGAN) and classifiers on the **train** split only.
- Model selection on **val**; final numbers on **test**.
- Synthetic images are added **only to the training set**, never to val or test.

## Known limits and biases
- Mostly **light skin tones** (HAM10000 comes mainly from Austrian and Australian clinics): results may not transfer to darker skin.
- **Several images per lesion** in HAM10000: images of the same lesion can appear in different splits, which can make real test images close to training images (important for the memorization check).
- Strong downsampling to 64 × 64 removes fine dermoscopic structures.
- Labels come from histopathology, follow-up or expert consensus, depending on the lesion; some label noise is possible.
- **Not for diagnosis:** this dataset and the models trained on it are for research and teaching only.
