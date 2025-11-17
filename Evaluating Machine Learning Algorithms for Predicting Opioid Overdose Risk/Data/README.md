# TEDS-D Data Exploration & Preprocessing
This notebook performs exploratory data analysis and preprocessing on the TEDS-D (Treatment Episode Data Set – Discharges) dataset provided by SAMHSA.

## Dataset Source
The full TEDS-D dataset can be downloaded from SAMHSA:
https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles?data_collection=1022

## What This Notebook Covers
This notebook focuses on cleaning and understanding the raw treatment episode records:
- Filtering the dataset to a specific geographic region
- Handling missing data and invalid entries
- Converting coded categorical variables into human-readable labels
- Generating basic exploratory visualizations:
  - Histograms
  - Frequency plots
  - Correlation heatmaps
- Preparing a clean, structured dataset for use in downstream modeling

This notebook serves as the foundation for the machine learning modeling workflow included in the primary notebook.
