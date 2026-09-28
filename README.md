# Khumbu Icefall Flood Early Warning System (TECH-405)

An exploratory data analysis (EDA), computer vision data pipeline, and machine learning framework for **Flood Early Warning from Satellite Images** of the **Khumbu Icefall** in the Sagarmatha (Mt. Everest) region of Nepal.

## Documentation & Explanations

- [Khumbu_Icefall_Flood_Early_Warning_Explanation.md](file:///d:/TECH%20project/Khumbu_Icefall_Flood_Early_Warning_Explanation.md): In-depth, paragraph-by-paragraph technical and environmental explanation of the dataset, quality audit, decadal visual similarity, train/val/test split rationale, and CNN feature engineering pipeline.
- [outputs/EDA_Report.md](file:///d:/TECH%20project/outputs/EDA_Report.md): Concise section-by-section engineering report with metrics and figures.

## Key Notebook & Artifacts

- [eda.ipynb](file:///d:/TECH%20project/eda.ipynb): Top-to-bottom executable Jupyter Notebook for data auditing, statistical analysis, visualization, and preprocessing.
- `dataset/`: 61 chronological satellite captures spanning 1958 through 2021.
- `outputs/figures/`: Generated visual figures (decade distributions, image dimensions, pixel histograms, decade-mean imagery, cosine similarity heatmaps, and temporal splits).
- `outputs/split.csv`: Chronologically partitioned splits (70% train, 15% validation, 15% test).
- `outputs/eda_summary.json`: Precomputed channel mean and standard deviation parameters for inference normalization.