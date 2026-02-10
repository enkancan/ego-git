# Ankara EGO Bus & Rail System Analysis

**Fragmented City, Disconnected Network: The Structural Misalignment between Urban Sprawl and Bus Services in Ankara**

## Overview

A comprehensive spatial and temporal analysis of Ankara's EGO public transit system. The study examines service-demand mismatches, route efficiency, and structural patterns using grid-based spatial analysis, anomaly detection, and regime shift analysis.

## Key Findings

- **233 days** of bus data analyzed (Dec 2023 – Oct 2024)
- **315 days** of metro/rail data (Dec 2023 – Nov 2024)
- **550 bus routes** + **5 rail lines** covering **1,382 grid cells** (1km × 1km)
- **13 critical grids** identified (both anomalous and unstable)
- **417 undersupply hotspots** and **353 oversupply coldspots** (LISA)
- Weekday vs Weekend: **-28.8% passengers**, **-21% occupancy**

## Repository Structure

```
ego-git/
├── README.md
├── .gitignore
│
├── data/                              # Processed datasets
│   ├── ego_data_with_dates_CORRECTED.csv      # Bus data (233 days)
│   ├── ego_metro_data_with_dates.csv          # Metro/rail data (315 days)
│   ├── ego_metro_data.csv                     # Raw metro data
│   ├── ego_route_stops.csv                    # Route-stop mapping
│   ├── ego_route_stops_all_coords.csv         # Stop coordinates
│   ├── normalized_all_transit.csv             # Metro + bus normalized
│   ├── daily_grid_data.csv                    # Daily grid (connectivity-weighted)
│   ├── daily_grid_data_position.csv           # Daily grid (position-weighted)
│   ├── grid_analysis.csv                      # Grid statistics
│   ├── grid_anomaly_results.csv               # Anomaly detection results
│   ├── grid_combined_analysis.csv             # Combined anomaly + regime shift
│   ├── grid_regime_shift_results.csv          # Regime shift results
│   ├── weekday_weekend_comparison.csv         # Weekday/weekend patterns
│   ├── kpi_tail_analysis.csv                  # KPI tail analysis
│   └── spatial_hotspot_analysis.csv           # LISA spatial clusters
│
├── notebooks/                         # Jupyter analysis notebooks
│   ├── Ego_Quick_Analysis.ipynb               # Quick exploratory analysis
│   └── Ego_Advanced_Analysis.ipynb            # Advanced analysis
│
├── scripts/                           # Data processing & visualization
│   ├── parse_metro_pdfs.py                    # Extract metro data from PDFs
│   ├── reparse_with_real_dates.py             # PDF date extraction
│   ├── normalize_all_transit.py               # Normalize metro + bus data
│   ├── create_grid_heatmap.py                 # Grid heatmaps (Folium)
│   ├── create_grid_by_metrics.py              # Multi-metric grid maps
│   ├── create_grid_occupancy_bus.py           # Bus occupancy grids
│   ├── create_square_grid_map.py              # Square grid maps
│   ├── visualize_raw_features_grid.py         # Raw feature visualization
│   ├── compare_distribution_methods.py        # Distribution comparison
│   ├── visualize_distribution_difference.py   # Distribution diff maps
│   └── html_to_png.py                         # HTML → PNG converter
│
├── ai_analysis/                       # ML-based analysis
│   ├── scripts/
│   │   ├── create_daily_grid_data.py          # Grid data (connectivity)
│   │   ├── create_daily_grid_data_position.py # Grid data (position)
│   │   ├── grid_anomaly_and_regime_analysis.py# Anomaly + regime shift
│   │   ├── visualize_anomaly_regime_results.py# Result visualization
│   │   └── advanced_validation_analysis.py    # LISA, KPI validation
│   ├── data/                          # Analysis results
│   └── outputs/                       # Interactive charts & maps
│       └── png/
│
└── maps/                              # Interactive maps
    ├── grid_*.html                    # Grid-based feature maps
    ├── heatmap_*.html                 # Heatmaps
    ├── interactive_*.html             # Plotly charts
    └── png/                           # Static PNG screenshots
```

## Methodology

### Data Pipeline
1. **PDF Parsing** → Extract daily bus & metro data from EGO PDF reports
2. **Date Correction** → Match reports to actual calendar dates
3. **Grid Aggregation** → 1km × 1km spatial grid, two distribution methods:
   - **Connectivity-weighted**: Passengers distributed by stop transfer power
   - **Position-weighted**: Terminal-biased distribution (30% first/last stops)

### Analysis Methods
- **Anomaly Detection**: Autoencoder (PCA), Isolation Forest, LOF, Graph-based
- **Regime Shift**: Embedding stability, Daily clustering
- **Spatial Analysis**: Global/Local Moran's I (LISA)
- **Validation**: KPI tail analysis, Weekday/weekend segmentation

## Tech Stack

- **Python 3.13** — pandas, numpy, scikit-learn
- **Visualization** — Matplotlib, Seaborn, Plotly, Folium
- **Spatial** — PySAL (LISA/Moran's I), NetworkX
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

## License

Academic research use.
