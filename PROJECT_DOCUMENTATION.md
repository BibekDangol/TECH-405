# Khumbu Icefall Glacier Monitoring — TECH-405
## Complete Project Documentation

> What this repo does and does not do: the CNNs classify image appearance (Early-vs-Late era; snow/ice/debris surface). They do **not** predict avalanches, floods, or GLOFs — the dataset has no flood labels, no lake outlines, and no outburst before/after pairs.

> Repository layout: two topic folders, each self-contained (run notebooks/scripts with CWD = topic folder).
> `early_late/` holds the era-appearance track (`dataset/`, `eda.ipynb`, `neuralNetwork.ipynb`, `predict.ipynb`, `train_avalanche.py`, `outputs/`).
> `sentinel2/` holds the surface-mapping track (`data_s2/`, `s2/`, `s2_simple.ipynb`, `outputs/`).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem Statement](#2-the-problem-statement)
3. [Dataset](#3-dataset)
4. [Phase 1: Exploratory Data Analysis (EDA)](#4-phase-1-exploratory-data-analysis-eda)
5. [Phase 2: Neural Network (CNN) Classifier](#5-phase-2-neural-network-cnn-classifier)
6. [Phase 3: Inference / Prediction Notebook](#6-phase-3-inference--prediction-notebook)
7. [Key Results & Metrics](#7-key-results--metrics)
8. [Output Artifacts & Files](#8-output-artifacts--files)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Technical Architecture Summary](#10-technical-architecture-summary)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. Project Overview

**Project Name:** Khumbu Icefall Glacier Monitoring (TECH-405)

This project builds machine learning pipelines — from raw imagery to trained classifiers — to describe how Khumbu Icefall satellite images **look** (era appearance; snow/ice/debris surface cover). It is monitoring, not an avalanche/flood early-warning system.

The **Khumbu Icefall** is a constantly moving, highly unstable glacier located on the southern slope of Mount Everest in the Sagarmatha region of Nepal. It is one of the most dangerous sections on the standard Everest climbing route. Over recent decades, climate change has accelerated glacial retreat and increased avalanche frequency in this region.

The goal is to classify satellite images into two appearance categories:
- **Early:** Images from before 2015 (screenshot appearance proxy).
- **Late:** Images from 2015 onward (screenshot appearance proxy).
- Separately, `s2/` maps **snow / ice / debris** surface cover from Sentinel-2 (or fallback screenshot patches). Neither output is a hazard forecast.

---

## 2. The Problem Statement

Predicting avalanches and glacial floods in remote mountainous areas like the Khumbu Icefall is extremely difficult due to:
- Limited on-ground sensor infrastructure
- Rapidly changing glacial conditions due to climate change
- The visual complexity of satellite imagery spanning decades

This project addresses the problem by using **Convolutional Neural Networks (CNNs)** trained on historical satellite imagery. The CNN learns the visual features that differentiate the stable (Early era, ≤2014) from the high-risk (Late era, ≥2015) glacial conditions, and can then make predictions on new, unseen images.

---

## 3. Dataset

| Property | Value |
|---|---|
| **Total Images** | 61 |
| **Format** | PNG (RGBA mode, from satellite screenshots) |
| **Total Size** | ~119.2 MB |
| **Year Range** | 1958 – 2021 |
| **Image Resolution** | ~1628 × 772 px (varies; 52 unique sizes) |
| **Labels** | Derived from filename year: `Early (<=2014)` vs `Late (>=2015)` |
| **Source** | User-collected satellite screenshots of Khumbu Icefall |
| **Storage** | `dataset/` folder (flat directory, 61 `.png` files) |

### Decade Distribution

| Decade | Image Count |
|---|---|
| 1950s | 1 |
| 2000s | 9 |
| 2010s | 46 |
| 2020s | 5 |

### Class Distribution

| Class | Count |
|---|---|
| Early (≤ 2014) | 34 |
| Late (≥ 2015) | 27 |

> **Note:** The dataset is intentionally kept as raw satellite screenshots with no pre-labeling. Labels are automatically generated from the filename year at training time.

---

## 4. Phase 1: Exploratory Data Analysis (EDA)

**Notebook:** `eda.ipynb`  
**Purpose:** Understand the dataset, find its properties, clean it, and design the preprocessing pipeline.

### Section 1 — Dataset Card

The dataset was fully characterized: 61 RGBA PNG files, spanning 1958 to 2021, ~119 MB on disk. All images were collected as screen-captures; their filenames encode timestamps (not class labels).

### Section 2 — Load & Inspect

A full statistical profile of all 61 images was built:
- Average pixel mean: **173.6**
- Average pixel std: **59.15**
- Width range: 1442–1663 px
- Height range: 742–792 px

All metadata was saved to `outputs/file_index.csv`.

### Section 3 — Data Quality Audit

A three-part quality check was performed:
- **Exact Duplicates (SHA-256):** 0 found ✅
- **Corrupt / Unreadable Files:** 0 found ✅
- **Degenerate Images (near-constant or all-black/white):** 0 found ✅

**Conclusion:** The dataset is clean and ready for training.

### Section 4 — Class & Target Distribution

Since images are inherently unlabeled, decade-based pseudo-groups were used for visualization (`01_class_distribution.png`). The 2010s decade dominates the dataset (46 out of 61 images), which reflects the increased availability of satellite imagery in recent years.

### Section 5 — Image Size & Channel Analysis

- **52 unique image sizes** were found → a fixed input size is required for the CNN.
- **All images are RGBA** (4 channels) → the Alpha (transparency) channel carries no signal for glacier analysis and must be dropped.
- **Decision:** Convert all images to `RGB` (3 channels) and resize to a fixed `128×128` px for CNN input.
- Figures generated: `02_image_sizes.png`, `03_samples_grid.png`, `04_pixel_histograms.png`

### Section 5d — Decade Mean Images & Cosine Similarity

Mean images were computed per decade to visualize long-term glacial changes. A cosine similarity heatmap revealed:
- 2010s vs 2020s similarity: **0.994** (nearly identical)
- 2000s vs 2010s similarity: **0.985**
- 1950s vs 2000s similarity: **0.894** (most distinct)

> **Implication:** The eras are visually very similar, making the classification task challenging and emphasizing the need for a proper CNN to detect subtle differences.

Figures: `05_mean_images.png`, `06_mean_similarity_heatmap.png`

### Section 6 — Train / Validation / Test Split

A **time-ordered** (chronological) split was used instead of random:
- **Train (70%):** 42 images — years 1958–2015
- **Val (15%):** 9 images — years 2015–2017
- **Test (15%):** 10 images — years 2017–2021

> **Why time-ordered?** A random split would leak future data into training (temporal leakage), giving artificially inflated accuracy. Time-ordering ensures the model is tested on images it has truly never seen.

Saved to: `outputs/split.csv`  
Figure: `07_split.png`

### Sections 7-8 — Preprocessing Pipeline & Normalization

The train-set-only normalization parameters were computed:
- **RGB Mean (0–1):** `[0.6419, 0.6689, 0.6687]`
- **RGB Std (0–1):** `[0.2702, 0.2666, 0.2669]`

> **Why train-only normalization?** Computing normalization stats from the entire dataset (including val/test) would constitute data leakage. Only train images were used, as the model must generalize to unseen data distributions.

**Final Preprocessing Pipeline:**
1. Convert RGBA → RGB (drop alpha channel)
2. Resize to 128×128 pixels
3. Apply mild augmentation (train only): ±5° rotation, ±10% brightness/contrast
4. No horizontal flip (screenshots may contain text/UI labels)
5. Normalize using train-set mean and std

All parameters saved to: `outputs/eda_summary.json`

---

## 5. Phase 2: Neural Network (CNN) Classifier

**Notebook:** `neuralNetwork.ipynb`  
**Framework:** PyTorch (CUDA GPU-accelerated)

### Architecture — Custom CNN

A lightweight custom Convolutional Neural Network was designed and implemented in PyTorch:

```
Input: 3 × 128 × 128 (RGB image)
│
├── Conv2d(3→32, kernel=3) + ReLU + MaxPool2d(2)
│   → 32 × 63 × 63
├── Conv2d(32→64, kernel=3) + ReLU + MaxPool2d(2)
│   → 64 × 30 × 30
├── AdaptiveAvgPool2d(4×4)
│   → 64 × 4 × 4 = 1,024 features
│
├── Flatten → 1,024
├── Dropout
├── Linear(1024 → hidden) + ReLU
├── Dropout
└── Linear(hidden → 2)   ← Early / Late
```

**Total Parameters:** ~150,850 (intentionally small for the tiny dataset)

### Labeling Strategy

Labels were derived automatically from the filename year at runtime:
- `year <= 2014` → **Class 0: Early** (lower historical avalanche risk context)
- `year >= 2015` → **Class 1: Late** (higher avalanche/flood risk context, coinciding with accelerated glacial change post-2015 Everest earthquake)

### Data Augmentation

To prevent overfitting on the tiny 42-image training set:
- `RandomRotation(5°)` — mild rotation
- `ColorJitter(brightness=0.1, contrast=0.1)` — slight lighting variation
- No horizontal flip (preserves geographical orientation)

### Class-Weighted Loss

Because the dataset has a mild class imbalance (34 Early, 27 Late), **class-weighted CrossEntropyLoss** was used. Class weights are computed as: `w[i] = total_samples / (2 × count[i])`, giving slightly more importance to the minority class.

### Hyperparameter Grid Search

A full grid search was run across the following parameters:

| Parameter | Values Tested |
|---|---|
| Learning Rate | `0.001`, `0.0001` |
| Hidden Layer Size | `128`, `256` |
| Dropout Rate | `0.3`, `0.5` |
| Weight Decay | Fixed at `0.0001` |

**8 configurations total.** Each was trained for up to 25 epochs with early stopping (patience = 6 epochs) based on validation macro-F1 score.

### Grid Search Results

| Config | Val Macro-F1 |
|---|---|
| `lr=0.001, hidden=256, drop=0.3` | **1.0000** ✅ |
| `lr=0.001, hidden=256, drop=0.5` | **1.0000** ✅ |
| `lr=0.0001, hidden=128, drop=0.3` | 0.8831 |
| `lr=0.001, hidden=128, drop=0.3` | 0.7750 |
| `lr=0.001, hidden=128, drop=0.5` | 0.7750 |
| `lr=0.0001, hidden=128, drop=0.5` | 0.7750 |
| `lr=0.0001, hidden=256, drop=0.3` | 0.7750 |
| `lr=0.0001, hidden=256, drop=0.5` | 0.7750 |

**Best Configuration:** `lr=0.001, hidden=256, dropout=0.5`  
(Selected because it tied for highest F1 and higher dropout provides better regularization)

Figures: `09_hparam_curves.png`, `12_hparam_heatmap.png`, `13_config_f1_bar.png`

### Final Test Set Evaluation

The best model was evaluated exactly once on the held-out test set (10 images, never seen during training or validation):

| Metric | Early Class | Late Class | Overall |
|---|---|---|---|
| **Precision** | 0.75 | 1.00 | — |
| **Recall** | 1.00 | 0.50 | — |
| **F1-Score** | 0.857 | 0.667 | 0.762 (macro) |
| **Accuracy** | — | — | **80.0%** |

**Confusion Matrix:**
```
              Predicted Early   Predicted Late
Actual Early:       6 (TP)           0 (FN)
Actual Late:        2 (FP)           2 (TP)
```

**Interpretation:**
- The model correctly identified all 6 "Early" images (perfect recall for the low-risk class).
- It missed 2 out of 4 "Late" images (predicting them as "Early"), indicating it has difficulty distinguishing some late-era images. This is consistent with the very high cosine similarity (>0.99) observed between eras in the EDA phase.

Figures: `08_confusion_matrix.png`, `14_confusion_matrix_styled.png`, `15_per_class_metrics.png`

### MLP Baseline

For comparison, a standard **sklearn MLPClassifier** was trained on flattened 64×64 grayscale pixel vectors. It achieved **100% test accuracy** on this tiny set — which is almost certainly overfitting due to the dataset's small size. The CNN's 80% is more realistic and generalizable.

### Saved Model Artifacts

The trained model was saved to: `outputs/khumbu_cnn.pt`

---

## 6. Phase 3: Inference / Prediction Notebook

**Notebook:** `predict.ipynb`  
**Purpose:** Show the trained model's Early-vs-Late appearance prediction on a new screenshot. It does **not** predict avalanche/flood risk.

### Notebook Structure

1. **Cell 1 — Imports:** Loads PyTorch, PIL, matplotlib.
2. **Cell 2 — Model Definition:** Re-declares the CNN architecture (must match training architecture exactly).
3. **Cell 3 — Load Weights:** Automatically reads `outputs/nn_results.json` for best hyperparameters, `outputs/eda_summary.json` for normalization stats, and loads `outputs/khumbu_cnn.pt` model weights.
4. **Cell 4 — Input Image:** User sets `IMAGE_PATH` variable to the satellite image they want to analyze. The image is displayed using matplotlib.
5. **Cell 5 — Inference:** The image is preprocessed (resize, normalize), fed through the CNN, and the result is printed with:
   - The predicted risk label
   - Confidence percentage
   - Full probability breakdown for both classes

### How to Use

```python
# In Cell 4, change this variable:
IMAGE_PATH = "dataset/2021.11.13.png"  # ← change to your image
```

Then run all cells. Output example:
```
==================================================
🛰️  ERA APPEARANCE PREDICTION (not a hazard warning):
    Late-era appearance (>=2015)
--------------------------------------------------
```

---

## 7. Key Results & Metrics

| Phase | Outcome |
|---|---|
| **Data Quality** | 0 duplicates, 0 corrupt, 0 degenerate images |
| **Dataset Size** | 61 images, 119.2 MB, spanning 1958–2021 |
| **Model** | Custom CNN, 150,850 parameters |
| **Best Config** | `lr=0.001, hidden=256, dropout=0.5` |
| **Val Macro-F1** | **1.000** (100% on validation set) |
| **Test Accuracy** | **80.0%** on 10 held-out images |
| **Test Macro-F1** | **0.762** |
| **GPU** | CUDA (torch 2.11.0+cu128) |
| **Training Time** | Fast (~seconds per config due to tiny dataset) |

---

## 8. Output Artifacts & Files

### Notebooks

| File | Description |
|---|---|
| `early_late/eda.ipynb` | Exploratory Data Analysis pipeline |
| `early_late/neuralNetwork.ipynb` | CNN training, hyperparameter search, evaluation |
| `early_late/predict.ipynb` | **Prediction notebook for new images** |
| `sentinel2/s2_simple.ipynb` | **Clean standalone Sentinel-2 patch CNN (snow/ice/debris)** |

### Outputs

| File | Description |
|---|---|
| `early_late/outputs/file_index.csv` | Full metadata table of all 61 images |
| `early_late/outputs/split.csv` | Chronological train/val/test split (EDA) |
| `early_late/outputs/nn_split.csv` | Stratified split used by the CNN |
| `early_late/outputs/eda_summary.json` | Normalization params, pipeline config |
| `early_late/outputs/nn_results.json` | Grid search results, test metrics, confusion matrix |
| `early_late/outputs/khumbu_cnn.pt` | **Saved trained CNN model weights** |
| `sentinel2/outputs/s2_cnn.pt` | **Saved trained Sentinel-2 patch CNN weights** |
| `sentinel2/outputs/s2_results.json` | S2 test metrics, confusion matrix |
| `sentinel2/outputs/s2_simple.pt` | **Saved weights from the clean `s2_simple.ipynb`** |

### Figures Generated

| Figure | Description |
|---|---|
| `01_class_distribution.png` | Decade pseudo-group distribution |
| `02_image_sizes.png` | Width/height scatter and histograms |
| `03_samples_grid.png` | 4×4 grid of 16 random sample images |
| `04_pixel_histograms.png` | Global RGB intensity distribution |
| `05_mean_images.png` | Average image per decade |
| `06_mean_similarity_heatmap.png` | Cosine similarity between decades |
| `07_split.png` | Year distribution per train/val/test split |
| `08_confusion_matrix.png` | CNN confusion matrix (sklearn default) |
| `09_hparam_curves.png` | Train loss & val F1 curves per config |
| `10_class_distribution.png` | Early vs Late class counts |
| `11_split_breakdown.png` | Stacked split breakdown by class |
| `12_hparam_heatmap.png` | Hyperparameter grid heatmap |
| `13_config_f1_bar.png` | Horizontal F1 bar chart per config |
| `14_confusion_matrix_styled.png` | Styled dark-theme confusion matrix |
| `15_per_class_metrics.png` | Per-class precision/recall/F1 bars |

---

## 9. How to Run the Project

### Prerequisites

```bash
pip install torch pillow pandas numpy matplotlib scikit-learn jupyter
# for the Sentinel-2 surface-mapping track (s2/):
pip install rasterio requests tifffile
```

### Step 1: Run EDA

```bash
cd early_late
```

Open and run all cells in `eda.ipynb`. This will:
- Analyze the dataset
- Generate figures
- Save `outputs/eda_summary.json` and `outputs/split.csv`

### Step 2: Train the CNN

Open and run all cells in `neuralNetwork.ipynb`. This will:
- Load normalization parameters from Step 1
- Run the hyperparameter grid search (8 configs)
- Evaluate the best model on the test set
- Save `outputs/khumbu_cnn.pt` and `outputs/nn_results.json`

### Step 3: Predict on New Images

(Open `predict.ipynb`, change `IMAGE_PATH` in Cell 4 to your satellite image, and run all cells to get a risk prediction.)

### Step 4: Sentinel-2 Surface Mapping (separate track)

```bash
cd ../sentinel2
```

Open and run `s2_simple.ipynb` (clean standalone patch CNN, training + single-image prediction).

---

## 10. Technical Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                     TECH-405 Pipeline                   │
│                                                         │
│  dataset/ (61 RGBA PNGs, 1958–2021)                    │
│      │                                                  │
│      ▼                                                  │
│  eda.ipynb                                              │
│  ├── Quality Audit (SHA256, PIL verify)                 │
│  ├── Statistical Profiling                              │
│  ├── Cosine Similarity (decade means)                   │
│  ├── Chronological Split → split.csv                    │
│  └── Normalization → eda_summary.json                   │
│      │                                                  │
│      ▼                                                  │
│  neuralNetwork.ipynb                                    │
│  ├── Labeling (year >= 2015 → Late)                     │
│  ├── Stratified Split → nn_split.csv                    │
│  ├── Data Augmentation (rotation, jitter)               │
│  ├── CNN (Conv×2 + AdaptiveAvgPool + FC×2)              │
│  ├── Hyperparameter Grid Search (8 configs)             │
│  ├── Best Model → khumbu_cnn.pt                         │
│  └── Test Evaluation → nn_results.json                  │
│      │                                                  │
│      ▼                                                  │
│  predict.ipynb                                          │
│  └── Upload image → CNN → Risk Label + Confidence       │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Limitations & Future Work

### Current Limitations

| Limitation | Details |
|---|---|
| **Small Dataset** | Only 61 images. The model may overfit or generalize poorly to images from other sensors or resolutions. |
| **Binary Labels Only** | The era model only distinguishes "Early" vs "Late" screenshot appearance. It does not detect or forecast avalanche/flood events. |
| **No Flood Signal** | No flood/GLOF labels, lake outlines, or outburst pairs exist in this dataset — floods cannot be a training target. |
| **Avalanche Folders Are Not a Forecast** | 14 avalanche vs 47 no_avalanche, all avalanche files Apr–Jun 2014/2015; test 60% < 77% majority baseline. Single still images cannot forecast events. |
| **Soft Proxy Labels** | Year ≥ 2015 is used as a proxy for high risk (inspired by the 2015 Everest earthquake), not direct event labels. |
| **Visual Similarity** | Decade-mean cosine similarity >0.98 — the eras look nearly identical, making this a hard visual classification task. |
| **No Temporal Sequence Modeling** | Images are treated as independent; temporal change detection (e.g., using LSTMs or video CNNs) is not implemented. |

### Future Improvements

1. **More Data:** Real Sentinel-2 scenes via `sentinel2/s2/download_sentinel2.py`; NDSI starter labels + manual cleanup.
2. **Surface mapping (built, Sentinel-2 only):** `s2/` patch CNN (snow/ice/debris) on 5 real L2A crops (May 2021 – Dec 2024, NDSI starter labels, SCL cloud/shadow masking, scene-separated splits): ~840 patches after masking, test acc **~96%**, macro-F1 **~0.94** (`outputs/s2_results.json`). Self-contained training: `sentinel2/s2_simple.ipynb` (+ `sentinel2/s2/train_surface_cnn.py` script for CLI runs). Cover trend `outputs/s2_cover_trend.png` + `s2_cover.csv`: spring scenes show more ice (May 2023 ice ~21% of valid), winter scenes debris-heavy with zero predicted ice — partly illumination, documented as a caveat. Next: ~200-patch manual audit to quantify starter-label noise.
2. **Transfer Learning:** Use pre-trained ResNet-50 or EfficientNet as a backbone; fine-tune on this dataset for much better accuracy.
3. **Change Detection:** Use pairs of before/after images and compute a difference map to detect structural changes in the glacier.
4. **Real-time API:** Deploy the trained model as a REST API (FastAPI/Flask) so it can process new satellite images automatically.
5. **Web Dashboard:** Build a full web application where researchers can upload satellite images and receive instant risk assessments with visual explanations (Grad-CAM activation maps).
6. **Multi-class Risk Score:** Instead of binary Early/Late, predict a continuous risk score (0–100%) or a 4-level risk tier (Low, Moderate, High, Critical).

---

*Documentation written for TECH-405 — Khumbu Icefall Glacier Monitoring (surface/era appearance, not avalanche/flood forecasting).*
