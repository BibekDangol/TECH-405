# Khumbu Icefall Glacier Monitoring (TECH-405)

Exploratory analysis, CNN baselines, and **Sentinel-2 surface mapping (snow / ice / debris)** for the **Khumbu Icefall**, Sagarmatha (Mt. Everest) region.

> Scope honesty: nothing in this repo predicts avalanches or floods. The 61 screenshots have no flood labels, no lake outlines, and no outburst pairs. The old Early-vs-Late CNN classifies pre-2015 vs post-2015 screenshot appearance only; `predict.ipynb` no longer issues warnings. The avalanche folders (14 vs 47, all spring 2014/2015, 60% vs 77% majority baseline) are not a forecast. The supported CNN project is surface mapping in `s2/` + `s2_surface_mapping.ipynb`.

## Documentation & Explanations

- [Khumbu_Icefall_Flood_Early_Warning_Explanation.md](file:///d:/TECH%20project/Khumbu_Icefall_Flood_Early_Warning_Explanation.md): In-depth, paragraph-by-paragraph technical and environmental explanation of the dataset, quality audit, decadal visual similarity, train/val/test split rationale, and CNN feature engineering pipeline.
- [outputs/EDA_Report.md](file:///d:/TECH%20project/outputs/EDA_Report.md): Concise section-by-section engineering report with metrics and figures.

## Key Notebook & Artifacts

- [eda.ipynb](file:///d:/TECH%20project/eda.ipynb): Top-to-bottom executable Jupyter Notebook for data auditing, statistical analysis, visualization, and preprocessing.
- `dataset/`: 61 chronological satellite captures spanning 1958 through 2021.
- `s2/` + `s2_surface_mapping.ipynb`: Sentinel-2 surface-mapping pipeline (snow / ice / debris patch CNN, monitoring not forecasting).
- `outputs/figures/`: Generated visual figures (decade distributions, image dimensions, pixel histograms, decade-mean imagery, cosine similarity heatmaps, and temporal splits).
- `outputs/split.csv`: Chronologically partitioned splits (70% train, 15% validation, 15% test).
- `outputs/eda_summary.json`: Precomputed channel mean and standard deviation parameters for inference normalization.