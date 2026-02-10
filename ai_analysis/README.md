# AI Analysis — Algorithms & Outputs Guide

## Overview

This module performs **unsupervised anomaly detection** and **regime shift analysis** on Ankara's bus grid data to identify structural service-demand misalignments.

---

## Scripts & What They Do

### 1. `create_daily_grid_data.py`
- **Purpose**: Creates daily grid-level features from raw bus data
- **Method**: Connectivity-weighted passenger distribution
- **Input**: `data/ego_data_with_dates_CORRECTED.csv`, `data/ego_route_stops_all_coords.csv`
- **Output**: `data/daily_grid_data.csv` (318K records, 1382 grids × 233 days)

### 2. `create_daily_grid_data_position.py`
- **Purpose**: Same as above but with position-based passenger distribution
- **Method**: Terminal-biased distribution (30% weight to first/last stops)
- **Output**: `data/daily_grid_data_position.csv`

### 3. `grid_anomaly_and_regime_analysis.py` ⭐
- **Purpose**: Main analysis — anomaly detection + regime shift detection
- **Input**: `data/daily_grid_data.csv`
- **Output**: 3 CSV files + individual method maps
- **Details below** ↓

### 4. `visualize_anomaly_regime_results.py`
- **Purpose**: Create interactive maps and charts from analysis results
- **Input**: `data/grid_combined_analysis.csv`
- **Output**: 3 maps + 5 charts + 1 dashboard (all HTML)

### 5. `advanced_validation_analysis.py`
- **Purpose**: Validate anomaly results with independent methods
- **Input**: `data/daily_grid_data.csv`, `data/grid_anomaly_results.csv`
- **Output**: 4 CSV files

---

## Algorithms Used

### PART 1: Anomaly Detection (Service-Demand Mismatch)

All methods use 6 features per grid: `stop_count`, `yolcu_per_stop`, `sefer_per_stop`, `kapasite_per_stop`, `doluluk_orani`, `kapasite_kullanimi`

| # | Algorithm | How It Works | Output Map |
|---|-----------|-------------|------------|
| 1.1 | **PCA Autoencoder** | Compresses 6D → 3D via PCA, reconstructs back. High reconstruction error = anomaly (doesn't fit normal patterns) | `grid_anomaly_autoencoder.html` |
| 1.2 | **Isolation Forest** | Builds random trees. Anomalous points are isolated faster (fewer splits needed). `contamination=5%` | `grid_anomaly_isolation_forest.html` |
| 1.3 | **Local Outlier Factor (LOF)** | Compares each grid's local density to its k=20 nearest neighbors. LOF > 1 = sparser than neighbors = outlier | `grid_anomaly_lof.html` |
| 1.4 | **Graph-Based Spatial** | Builds spatial graph (1.5km threshold). Computes deviation from neighbor average. Grids different from their spatial neighbors = anomaly | `grid_anomaly_graph.html` |

**Consensus**: A grid is flagged as anomalous only if **≥2 out of 4 methods** agree.

→ **Result**: `data/grid_anomaly_results.csv` — 46 anomalous grids (3.3%)

### PART 2: Regime Shift Detection (Temporal Instability)

| # | Algorithm | How It Works | Output Map |
|---|-----------|-------------|------------|
| 2.1 | **Embedding Stability** | Creates daily K-Means embeddings (cluster assignments). Computes std of embeddings over time. High std = unstable identity | `grid_regime_embedding.html` |
| 2.2 | **Cluster Switching** | Fits global K-Means (k=5). Tracks how often each grid switches clusters day-to-day. High switch rate = regime shift | `grid_regime_clustering.html` |

**Consensus**: Either method flags instability → regime shift detected.

→ **Result**: `data/grid_regime_shift_results.csv` — 102 unstable grids (7.4%)

### PART 3: Combined Critical Grids

Grids that are **both anomalous AND unstable** = **critical**.

→ **Result**: `data/grid_combined_analysis.csv` — **13 critical grids**

---

## Validation Methods (advanced_validation_analysis.py)

| # | Method | Purpose | Output |
|---|--------|---------|--------|
| 1 | **Weekday/Weekend Segmentation** | Check if "anomalies" are just normal weekend patterns | `data/weekday_weekend_comparison.csv` |
| 2 | **KPI Tail Analysis** | Compare ML anomalies with simple statistical outliers (>p95 / <p5) | `data/kpi_tail_analysis.csv` |
| 3 | **LISA (Local Moran's I)** | Spatial autocorrelation — find HH/LL clusters (undersupply/oversupply hotspots) | `data/spatial_hotspot_analysis.csv` |
| 4 | **Change-Point Detection (PELT)** | Find structural breaks in time series per grid | `data/changepoint_analysis.csv` |

---

## Output Files

### Maps (HTML — Folium)

| File | Content | Algorithm |
|------|---------|-----------|
| `grid_anomaly_map.html` | All anomalous grids (consensus ≥2) colored by severity | PCA + IF + LOF + Graph consensus |
| `grid_regime_shift_map.html` | Unstable grids colored by instability score | Embedding + Clustering |
| `grid_critical_map.html` | 13 critical grids (anomalous + unstable) | Combined |

### Charts (HTML — Plotly)

| File | Content |
|------|---------|
| `anomaly_methods_comparison.html` | Venn-style comparison of 4 anomaly methods |
| `anomaly_score_distributions.html` | Score distributions for each method |
| `regime_shift_distribution.html` | Instability score histogram |
| `instability_vs_anomaly_scatter.html` | Scatter: anomaly count vs instability |
| `top20_critical_grids.html` | Bar chart of top 20 most critical grids |

### Dashboard

| File | Content |
|------|---------|
| `grid_analysis_dashboard.html` | Combined dashboard with links to all outputs |

### PNG Screenshots

All maps have PNG versions in `outputs/png/`.

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Total grids analyzed | 1,382 |
| Anomalous grids (consensus ≥2/4) | 46 (3.3%) |
| Unstable grids (regime shift) | 102 (7.4%) |
| **Critical grids (both)** | **13 (0.9%)** |
| Undersupply hotspots (LISA HH) | 417 |
| Oversupply coldspots (LISA LL) | 353 |
| Weekday/weekend passenger diff | -28.8% |
