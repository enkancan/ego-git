# Ankara EGO Bus & Rail System Analysis



## Overview

A comprehensive spatial and temporal analysis of Ankara's EGO public transit system. The study examines service-demand mismatches, route efficiency, and structural patterns using grid-based spatial analysis, anomaly detection, and regime shift analysis. 

## Key Findings

- **173 days** of bus data analyzed (Dec 2023 – Oct 2024)
- **315 days** of metro/rail data (Dec 2023 – Nov 2024)
- **550 bus routes** + **5 rail lines** covering **1,382 grid cells** (1km × 1km)
- **13 critical grids** identified (both anomalous and unstable)
- **417 undersupply hotspots** and **353 oversupply coldspots** (LISA)
- Weekday vs Weekend: **-28.8% passengers**, **-21% occupancy**


## Methodology

### Data Pipeline
1. **PDF Parsing** → Extract daily bus & metro data from EGO PDF reports
2. **Date Correction** → Match reports to actual calendar dates
3. **Grid Aggregation** → 1km × 1km spatial grid, two distribution methods:
   - **Connectivity-weighted**: Passengers distributed by stop transfer power
   - **Position-weighted**: Terminal-biased distribution (30% first/last stops)
  
<img width="1904" height="933" alt="grid_doluluk_orani" src="https://github.com/user-attachments/assets/362d60e1-2b98-420e-857e-62131bad6a1a" />



### Analysis Methods
- **Anomaly Detection**: Autoencoder (PCA), Isolation Forest, LOF, Graph-based
- **Regime Shift**: Embedding stability, Daily clustering
- **Spatial Analysis**: Global/Local Moran's I (LISA)


<img width="3249" height="1000" alt="6-grid" src="https://github.com/user-attachments/assets/c2947ecc-f734-4438-bdd8-c609150ad8e7" />




<img width="2957" height="1000" alt="category_radar_chart" src="https://github.com/user-attachments/assets/762b87bc-7743-4f1e-8222-18e4b2658eb2" />


  


## Tech Stack

- **Python 3.13** — pandas, numpy, scikit-learn
- **Visualization** — Matplotlib, Seaborn, Plotly, Folium





- **ML** — Scikit-learn (Isolation Forest, LOF, K-Means, PCA)

## Quick Start

```bash
pip install pandas numpy matplotlib seaborn folium plotly scikit-learn pysal networkx pdfplumber

# Create grid data
python ai_analysis/scripts/create_daily_grid_data.py

# Run anomaly analysis
python ai_analysis/scripts/grid_anomaly_and_regime_analysis.py

# Visualize results
python ai_analysis/scripts/visualize_anomaly_regime_results.py

# Generate grid maps
python scripts/visualize_raw_features_grid.py --input data/daily_grid_data.csv
```

## Data Sources

- **EGO Genel Müdürlüğü** — Ankara Metropolitan Municipality, Public Transit Authority
- Daily bus route reports & metro/rail passenger data (PDF)
- -Overpass and Google Geocoding API

## License

Academic research use.

## Citation

If you use this work, please cite:

```bibtex
@misc{kancan2026skeletaltrap,
  title        = {The Skeletal Trap: Mapping Spatial Inequality and Ghost Stops in Ankara's Transit Network},
  author       = {Kancan, Elifnaz},
  year         = {2026},
  eprint       = {2602.15470},
  archivePrefix= {arXiv},
  primaryClass = {physics.soc-ph},
  doi          = {10.48550/arXiv.2602.15470},
  url          = {https://arxiv.org/abs/2602.15470}
}
```
