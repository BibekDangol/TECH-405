"""Build course-report DOCX explaining the two Sentinel-2 notebooks."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = Path(__file__).resolve().parent
OUT = BASE / "Khumbu_Notebooks_Explained.docx"
FIG_SIMPLE_CONF = BASE / "outputs" / "s2_simple_confusion.png"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def h1(t): doc.add_heading(t.replace("—", "-"), level=1)
def h2(t): doc.add_heading(t.replace("—", "-"), level=2)
def h3(t): doc.add_heading(t.replace("—", "-"), level=3)
def para(t, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold = bold
    r.italic = italic
    return p
def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")

def add_table(headers, rows):
    tab = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tab.style = "Light Grid Accent 1"
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, hh in enumerate(headers):
        tab.rows[0].cells[j].text = hh
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            tab.rows[i].cells[j].text = str(val)
    doc.add_paragraph()

# ---------- Title ----------
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Land-Cover Classification of the Khumbu Icefall Region\nusing Sentinel-2 — Notebook Explanation")
r.bold = True
r.font.size = Pt(16)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("Course report covering Khumbu_Icefall_Sentinel2_Classification.ipynb "
            "(main) and s2_simple.ipynb (baseline)").italic = True

h1("Abstract")
para("This report explains two Jupyter notebooks that classify the Khumbu Icefall / Everest Base "
     "Camp area from Sentinel-2 imagery. The main notebook (Khumbu_Icefall_Sentinel2_Classification.ipynb) "
     "builds a cloud-masked Sentinel-2 L2A median composite for Oct–Dec 2021, takes ESA WorldCover 2021 as "
     "ground truth remapped to four classes (Vegetation, Bare/Rock, Snow/Ice, Water), engineers spectral "
     "indices, and compares a pixel-wise MLP against a patch-based CNN with a leak-free spatial block split. "
     "The supporting notebook (s2_simple.ipynb) classifies 64x64 RGB patches into snow / ice / debris with a "
     "small CNN, reaching 94.7% test accuracy (macro-F1 0.929). In the recorded Khumbu run both models collapsed "
     "to the majority Snow/Ice class (macro-F1 ~0.17, AUC ~0.5), so the report also discusses why the two "
     "notebooks differ in outcome and what to fix.")

h1("1. Study area and data sources")
h2("1.1 Khumbu Icefall notebook")
bullets([
    "Area: bounding box [86.84, 27.96, 86.96, 28.05] (lon/lat), icefall centre (86.899, 27.988); UTM 45N (EPSG:32645), 10 m grid.",
    "Imagery: Sentinel-2 L2A via Microsoft Planetary Computer STAC, 2021-10-01/2021-12-15, scene cloud cover < 20%; "
    "6 scenes found; cloud-masked (SCL classes 0,1,3,8,9,10) median composite of B02, B03, B04, B08, B11, B12.",
    "Ground truth: ESA WorldCover 2021 v200 (10 m), resampled to the same grid with nearest neighbour.",
    "Cache: composite stored in khumbu_s2_worldcover_cache.npz; trained weights in mlp_khumbu.pt / cnn_khumbu.pt.",
])
h2("1.2 s2_simple notebook (baseline)")
bullets([
    "Pre-cut 64x64 RGB patches listed in data_s2/patches.csv: train n=389, val n=186, test n=265; classes snow / ice / debris.",
    "Normalisation statistics computed on training pixels (mean ~[0.569, 0.559, 0.538], std ~[0.332, 0.342, 0.355]).",
    "Train augmentation: random horizontal flip + rotation in [-10, 10] degrees.",
])

h1("2. Label mapping (Khumbu)")
para("WorldCover codes are remapped to four target classes:")
add_table(["WorldCover code", "Meaning", "Target class"],
          [["10, 20, 30, 40, 90, 95, 100", "trees, shrub, grass, crop, wetland, mangrove, moss", "0 Vegetation"],
           ["50, 60", "built-up, bare / sparse vegetation", "1 Bare/Rock"],
           ["70", "snow and ice", "2 Snow/Ice"],
           ["80", "permanent water", "3 Water"]])
para("Pixels with no valid mapping (-1) or non-finite reflectance are excluded. Rare classes (< 500 pixels) "
     "are dropped and ids made contiguous before training.", italic=True)

h1("3. Methods — Khumbu notebook")
h2("3.1 Feature engineering")
para("Six surface-reflectance bands plus three normalised indices (9 channels total):")
bullets([
    "NDSI = (B03 - B11) / (B03 + B11) — snow (high green, low SWIR).",
    "NDVI = (B08 - B04) / (B08 + B04) — vegetation.",
    "NDWI = (B03 - B08) / (B03 + B08) — water.",
])
h2("3.2 Leak-free spatial split")
para("Neighbouring pixels are highly correlated, so a random pixel split would leak (especially for the CNN "
     "whose patches overlap). The scene is cut into 64x64 px blocks assigned 70/15/15 to train/val/test, "
     "then pixels are sampled per class with caps (train 20000, val 5000, test 10000 per class). Channel mean/std "
     "are computed on training pixels only. Seed 42.")
h2("3.3 Models")
add_table(["Model", "Input", "Architecture"],
          [["MLP (pixel-wise)", "(N, 9) single-pixel spectra",
            "Linear 9-128 + BN/ReLU/Dropout(0.2), 128-64, 64-32, 32-n_cls"],
           ["CNN (patch-based)", "(N, 9, 9, 9) centred windows",
            "Conv blocks 9-32, 32-64, MaxPool, 64-128, AdaptiveAvgPool, head Dropout(0.3) + Linear 128-64-n_cls"]])
para("Training: Adam (lr 1e-3, weight decay 1e-4), class-weighted cross-entropy, ReduceLROnPlateau (factor 0.5, "
     "patience 3), early stopping on validation macro-F1 (patience 8). Epochs: MLP 40, CNN 25 (batch 512 / 256). "
     "Metrics: accuracy, macro precision/recall/F1, macro AUC-ROC one-vs-rest, per-class report, confusion matrices, "
     "ROC curves, full-scene prediction maps + Khumbu Icefall zoom.")

h1("4. Methods — s2_simple notebook (brief)")
add_table(["Item", "Choice"],
          [["Model", "CNN: Conv 3-32 + ReLU + MaxPool, Conv 32-64 + ReLU + MaxPool, AdaptiveAvgPool(4x4), "
            "head Flatten + Dropout(0.3) + Linear(1024-128) + ReLU + Dropout(0.3) + Linear(128-3); ~150,979 params"],
           ["Optimiser/loss", "Adam lr 1e-3, weight decay 1e-4; class-weighted CE (weights [1.39, 5.4, 0.48] "
            "for train counts snow 93 / ice 24 / debris 272)"],
           ["Schedule", "5 epochs, batch 32; monitored val accuracy + macro-F1"],
           ["Artefacts", "outputs/s2_simple.pt, outputs/s2_simple_confusion.png"]])

h1("5. Results — Khumbu (recorded run)")
para("From sentinel2/results_summary.csv and results_per_class.csv:")
add_table(["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "AUC-ROC (macro OvR)"],
          [["MLP", "0.4990", "0.1248", "0.2500", "0.1665", "0.5000"],
           ["CNN", "0.5025", "0.3275", "0.2517", "0.1711", "0.5013"]])
add_table(["Class", "MLP Prec/Rec/F1", "CNN Prec/Rec/F1"],
          [["Vegetation", "0 / 0 / 0", "0 / 0 / 0"],
           ["Bare/Rock", "0 / 0 / 0", "0.809 / 0.009 / 0.018"],
           ["Snow/Ice", "0.499 / 1.0 / 0.666", "0.501 / 0.998 / 0.667"],
           ["Water", "0 / 0 / 0", "0 / 0 / 0"]])
para("Interpretation: both models collapsed to the majority Snow/Ice class (recall ~1.0 on Snow/Ice, ~0 elsewhere; "
     "AUC ~0.5 is random). The CNN's +0.005 F1 lead is noise, not a real win. Confusion is concentrated on "
     "Snow/Ice vs Bare/Rock — expected around debris-covered ice, shadowed slopes, and noisy WorldCover labels "
     "(debris-covered ice is often labelled bare ground). "
     "Note: Khumbu figures (RGB composite, label map, index maps, loss curves, confusion/ROC, scene maps) are "
     "generated at notebook run time and were not saved as image files in this checkout, so they are described "
     "rather than embedded here.", italic=True)

h1("6. Results — s2_simple")
bullets([
    "Epoch log: val accuracy 97.3 / 95.7 / 91.4 / 97.9 / 94.6%; val macro-F1 ending 0.7446 (best 0.8225 at epoch 4).",
    "Final: train 96.14%, val 94.62%, test 94.72%; macro avg precision 0.9205 / recall 0.9384 / F1 0.9290.",
    "Per class (test, n=265): snow P 0.9636 / R 0.9815 (n=54); ice P 0.8222 / R 0.8810 (n=42); "
    "debris P 0.9758 / R 0.9527 (n=169).",
    "Single-image demo (data_s2/patches/test/debris/2003.2.24_0379.png): predicted debris at 99.94% confidence.",
])
if FIG_SIMPLE_CONF.exists():
    h3("Figure: s2_simple test confusion matrix")
    doc.add_picture(str(FIG_SIMPLE_CONF), width=Inches(5.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para("Confusion matrix on the s2_simple test set (265 patches).", italic=True)
else:
    para("Figure outputs/s2_simple_confusion.png was not found; re-run s2_simple.ipynb to regenerate it.", italic=True)

h1("7. Comparison and discussion")
bullets([
    "Task difficulty: s2_simple classifies curated RGB patches (snow vs ice vs debris) with balanced, clean labels; "
    "Khumbu does dense per-pixel mapping over a whole mountain scene with severe imbalance and label noise.",
    "Why the CNN should win on Khumbu in principle: 9x9 spatial context (texture, crevasses, neighbours) on top of "
    "spectra + indices; the MLP sees only one pixel. The recorded run shows no meaningful gap only because neither "
    "model learned the minority classes.",
    "What to fix on Khumbu first: stronger imbalance handling (focal loss / oversampling / lower caps), more epochs "
    "or lower LR, check val-F1 curves for early-stopping behaviour, add terrain features (Copernicus DEM elevation/slope), "
    "consider a U-Net for dense prediction and multi-temporal inputs.",
    "AUC vs F1: per-class AUC can look high even when F1 is low since AUC measures ranking, not the decision threshold.",
])

h1("8. Limitations and reproducibility")
bullets([
    "WorldCover labels are a derived product (Sentinel-1/2 based), not field survey — treat as noisy reference.",
    "Single post-monsoon composite (Oct–Dec 2021); no terrain or multi-season information.",
    "Class caps change the test-sample prior; whole-scene accuracy includes training blocks — use test-block metrics.",
    "Reproduce Khumbu: set DATA_SOURCE=planetary_computer (needs internet), QUICK_TEST=1 for a smoke run; "
    "config SEED 42, PATCH 9, BLOCK 64, BANDS B02/B03/B04/B08/B11/B12, EPSG 32645, RES 10.",
    "Reproduce s2_simple: needs data_s2/patches.csv + patch PNGs; 5 epochs on CUDA/CPU; reload check outputs/s2_simple.pt.",
    "Source files: sentinel2/Khumbu_Icefall_Sentinel2_Classification.ipynb (+ .bak.ipynb with original narrative), "
    "sentinel2/s2_simple.ipynb, results_summary.csv, results_per_class.csv.",
])

doc.save(OUT)
print(f"saved {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
