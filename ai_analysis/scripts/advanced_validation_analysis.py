"""
ADVANCED VALIDATION ANALYSIS
=============================

4 ileri seviye validasyon tekniği:

1. WEEKDAY/WEEKEND SEGMENTATION
   - Hafta içi vs hafta sonu ayrı normlar
   - "Anomali" sandığımız şey normal hafta sonu paterni mi?

2. SIMPLE KPI TAIL ANALYSIS
   - Tek metriklerle (kapasite_kullanimi, yolcu_per_stop, sefer_per_stop) tail analizi
   - ML skorlarını basit KPI threshold'larıyla doğrula

3. SPATIAL HOTSPOT TESTING (Moran's I / LISA)
   - Undersupply/oversupply kümeleri bul
   - Bölgesel kümeler > tekil outlier (daha güvenilir müdahale hedefi)

4. TIME SERIES CHANGE-POINT DETECTION (PELT/Ruptures)
   - Grid bazında günlük "mismatch endeksi" (yolcu/kapasite)
   - Kırılma noktalarını bul → rejim kaymasını somut tarih/olayla eşle
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED VALIDATION ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 0: LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("PART 0: LOADING DATA")
print("=" * 80)

df = pd.read_csv('data/daily_grid_data.csv')
df['tarih'] = pd.to_datetime(df['tarih'])

# Add day of week
df['day_of_week'] = df['tarih'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday

print(f"\nTotal records: {len(df):,}")
print(f"Date range: {df['tarih'].min()} to {df['tarih'].max()}")
print(f"Unique grids: {df['grid_id'].nunique()}")
print(f"Weekday records: {len(df[~df['is_weekend']]):,}")
print(f"Weekend records: {len(df[df['is_weekend']]):,}")

if len(df[df['is_weekend']]) == 0:
    print("\n*** WARNING: No weekend data found! ***")
    print("    EGO PDF reports are only published for weekdays.")
    print("    Weekday/weekend segmentation analysis will be skipped.")

# ============================================================================
# PART 1: WEEKDAY/WEEKEND SEGMENTATION
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: WEEKDAY/WEEKEND SEGMENTATION")
print("=" * 80)

if len(df[df['is_weekend']]) == 0:
    print("\nSkipping weekday/weekend segmentation (no weekend data available)")
    print("Alternative: Analyzing day-of-week patterns (Monday-Friday)...")

    # Alternative: analyze by day of week instead
    dow_stats = df.groupby('day_of_week').agg({
        'yolcu_per_stop': 'mean',
        'sefer_per_stop': 'mean',
        'doluluk_orani': 'mean',
        'kapasite_kullanimi': 'mean'
    })

    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    dow_stats.index = [day_names[i] for i in dow_stats.index]

    print("\nDay-of-week patterns:")
    print(dow_stats.round(2))

    # Save
    dow_stats.to_csv('data/day_of_week_patterns.csv')
    print("\n[SAVED] data/day_of_week_patterns.csv")

else:
    print("\nAnalyzing patterns by day type...")

    # Aggregate by grid and day type
    df_weekday = df[~df['is_weekend']].groupby('grid_id').agg({
        'yolcu_per_stop': 'mean',
        'sefer_per_stop': 'mean',
        'kapasite_per_stop': 'mean',
        'doluluk_orani': 'mean',
        'kapasite_kullanimi': 'mean',
        'stop_count': 'mean'
    }).reset_index()

    df_weekend = df[df['is_weekend']].groupby('grid_id').agg({
        'yolcu_per_stop': 'mean',
        'sefer_per_stop': 'mean',
        'kapasite_per_stop': 'mean',
        'doluluk_orani': 'mean',
        'kapasite_kullanimi': 'mean',
        'stop_count': 'mean'
    }).reset_index()

    # Compare distributions
    print(f"\nWeekday vs Weekend Comparison:")
    print(f"{'Metric':<25} {'Weekday Mean':>15} {'Weekend Mean':>15} {'% Change':>12}")
    print("-" * 70)

    metrics = ['yolcu_per_stop', 'sefer_per_stop', 'kapasite_per_stop', 'doluluk_orani', 'kapasite_kullanimi']
    for metric in metrics:
        weekday_mean = df_weekday[metric].mean()
        weekend_mean = df_weekend[metric].mean()
        pct_change = ((weekend_mean - weekday_mean) / weekday_mean) * 100 if weekday_mean != 0 else 0
        print(f"{metric:<25} {weekday_mean:>15.2f} {weekend_mean:>15.2f} {pct_change:>11.1f}%")

    # Detect grids with strong weekday/weekend difference
    print("\n\nDetecting grids with significant weekday/weekend pattern changes...")

    df_compare = df_weekday.merge(df_weekend, on='grid_id', suffixes=('_weekday', '_weekend'))

    # Calculate relative change in capacity utilization
    df_compare['util_change'] = abs(df_compare['kapasite_kullanimi_weekend'] - df_compare['kapasite_kullanimi_weekday'])

    # Flag grids with >30% change
    threshold_change = 30
    df_compare['high_weekend_variance'] = df_compare['util_change'] > threshold_change

    print(f"Grids with >{threshold_change}% weekday/weekend difference: {df_compare['high_weekend_variance'].sum()}")

    # Save
    df_compare.to_csv('data/weekday_weekend_comparison.csv', index=False)
    print("\n[SAVED] data/weekday_weekend_comparison.csv")

# ============================================================================
# PART 2: SIMPLE KPI TAIL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: SIMPLE KPI TAIL ANALYSIS")
print("=" * 80)

print("\nPerforming tail analysis on individual KPIs...")

# Aggregate to grid level
df_grid = df.groupby('grid_id').agg({
    'grid_lat': 'first',
    'grid_lon': 'first',
    'yolcu_per_stop': 'mean',
    'sefer_per_stop': 'mean',
    'kapasite_per_stop': 'mean',
    'doluluk_orani': 'mean',
    'kapasite_kullanimi': 'mean',
    'stop_count': 'mean'
}).reset_index()

# Define tail thresholds (top/bottom 5%)
tail_percentile = 95

kpis = {
    'kapasite_kullanimi': 'Capacity Utilization',
    'yolcu_per_stop': 'Passengers per Stop',
    'sefer_per_stop': 'Trips per Stop',
    'doluluk_orani': 'Occupancy Rate'
}

tail_results = {}

print(f"\nTail Analysis (>{tail_percentile}th percentile or <{100-tail_percentile}th):")
print(f"{'KPI':<30} {'High Tail (>p95)':>20} {'Low Tail (<p5)':>20}")
print("-" * 75)

for kpi, name in kpis.items():
    p5 = np.percentile(df_grid[kpi], 100 - tail_percentile)
    p95 = np.percentile(df_grid[kpi], tail_percentile)

    high_tail = df_grid[df_grid[kpi] > p95]['grid_id'].tolist()
    low_tail = df_grid[df_grid[kpi] < p5]['grid_id'].tolist()

    tail_results[kpi] = {
        'high_tail': high_tail,
        'low_tail': low_tail,
        'p5': p5,
        'p95': p95
    }

    print(f"{name:<30} {len(high_tail):>20} {len(low_tail):>20}")

# Cross-validate with ML anomaly results
print("\n\nCross-validation with ML Anomaly Results:")

# Load ML results
df_anomaly = pd.read_csv('data/grid_anomaly_results.csv', encoding='utf-8-sig')
ml_anomalies = set(df_anomaly[df_anomaly['anomaly_count'] >= 2]['grid_id'].tolist())

# Check overlap
print(f"\nML detected anomalies (consensus >=2): {len(ml_anomalies)}")

for kpi, name in kpis.items():
    high_tail_set = set(tail_results[kpi]['high_tail'])
    low_tail_set = set(tail_results[kpi]['low_tail'])
    tail_union = high_tail_set | low_tail_set

    overlap = ml_anomalies & tail_union
    overlap_pct = (len(overlap) / len(ml_anomalies)) * 100 if len(ml_anomalies) > 0 else 0

    print(f"\n{name}:")
    print(f"  KPI tail grids: {len(tail_union)}")
    print(f"  Overlap with ML anomalies: {len(overlap)} ({overlap_pct:.1f}%)")
    print(f"  ML-only (not in KPI tail): {len(ml_anomalies - tail_union)}")
    print(f"  KPI-only (not in ML): {len(tail_union - ml_anomalies)}")

# Save tail analysis results
df_grid['in_capacity_util_tail'] = df_grid['grid_id'].isin(tail_results['kapasite_kullanimi']['high_tail'] + tail_results['kapasite_kullanimi']['low_tail'])
df_grid['in_passenger_tail'] = df_grid['grid_id'].isin(tail_results['yolcu_per_stop']['high_tail'] + tail_results['yolcu_per_stop']['low_tail'])
df_grid.to_csv('data/kpi_tail_analysis.csv', index=False)
print("\n[SAVED] data/kpi_tail_analysis.csv")

# ============================================================================
# PART 3: SPATIAL HOTSPOT TESTING (Moran's I / LISA)
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: SPATIAL HOTSPOT TESTING")
print("=" * 80)

try:
    from libpysal import weights
    from esda.moran import Moran, Moran_Local

    print("\nComputing spatial autocorrelation (Moran's I)...")

    # Create spatial weights matrix (k-nearest neighbors)
    coords = df_grid[['grid_lon', 'grid_lat']].values
    k = 8  # 8 nearest neighbors

    w = weights.KNN.from_array(coords, k=k)
    w.transform = 'r'  # Row-standardized

    print(f"Spatial weights matrix: {w.n} grids, k={k} neighbors")

    # Compute Moran's I for each metric
    print(f"\nGlobal Moran's I (spatial autocorrelation):")
    print(f"{'Metric':<30} {'Moran I':>12} {'p-value':>12} {'Interpretation':<20}")
    print("-" * 80)

    moran_results = {}

    for kpi, name in kpis.items():
        y = df_grid[kpi].values
        moran = Moran(y, w, permutations=999)

        moran_results[kpi] = moran

        # Interpretation
        if moran.p_sim < 0.05:
            if moran.I > 0:
                interp = "Clustered (similar)"
            else:
                interp = "Dispersed (dissimilar)"
        else:
            interp = "Random"

        print(f"{name:<30} {moran.I:>12.4f} {moran.p_sim:>12.4f} {interp:<20}")

    # Local Moran's I (LISA) - identify hotspots
    print("\n\nLocal Indicators of Spatial Association (LISA):")

    # Focus on capacity utilization
    kpi = 'kapasite_kullanimi'
    y = df_grid[kpi].values
    lisa = Moran_Local(y, w, permutations=999)

    # Classify into quadrants
    # 1 = HH (High-High), 2 = LH (Low-High), 3 = LL (Low-Low), 4 = HL (High-Low)
    df_grid['lisa_quadrant'] = lisa.q
    df_grid['lisa_pvalue'] = lisa.p_sim
    df_grid['lisa_significant'] = lisa.p_sim < 0.05

    # Count clusters
    quadrant_names = {1: 'HH (High-High)', 2: 'LH (Low-High)', 3: 'LL (Low-Low)', 4: 'HL (High-Low)'}

    print(f"\nLISA Clusters (Capacity Utilization, p<0.05):")
    for q, name in quadrant_names.items():
        count = len(df_grid[(df_grid['lisa_quadrant'] == q) & (df_grid['lisa_significant'])])
        print(f"  {name}: {count} grids")

    # Identify hotspots (HH) and coldspots (LL)
    hotspots = df_grid[(df_grid['lisa_quadrant'] == 1) & (df_grid['lisa_significant'])]['grid_id'].tolist()
    coldspots = df_grid[(df_grid['lisa_quadrant'] == 3) & (df_grid['lisa_significant'])]['grid_id'].tolist()

    print(f"\nUndersupply Clusters (HH - high capacity util + high neighbors): {len(hotspots)}")
    print(f"Oversupply Clusters (LL - low capacity util + low neighbors): {len(coldspots)}")

    # Save
    df_grid.to_csv('data/spatial_hotspot_analysis.csv', index=False)
    print("\n[SAVED] data/spatial_hotspot_analysis.csv")

except ImportError:
    print("\n[WARNING] libpysal not installed. Skipping spatial analysis.")
    print("Install with: pip install libpysal esda")

# ============================================================================
# PART 4: TIME SERIES CHANGE-POINT DETECTION (PELT)
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: TIME SERIES CHANGE-POINT DETECTION")
print("=" * 80)

try:
    import ruptures as rpt

    print("\nDetecting structural breaks in mismatch index...")

    # Create mismatch index: passenger/capacity ratio
    df['mismatch_index'] = df['yolcu_per_stop'] / df['kapasite_per_stop'].replace(0, 1)

    # Focus on grids with enough temporal data
    grid_counts = df.groupby('grid_id')['tarih'].count()
    valid_grids = grid_counts[grid_counts >= 100].index.tolist()

    print(f"Analyzing {len(valid_grids)} grids with >=100 days of data")

    # Detect change-points for each grid
    changepoint_results = []

    # Sample: analyze top 50 grids by variance (most likely to have breaks)
    grid_variances = df.groupby('grid_id')['mismatch_index'].var().sort_values(ascending=False)
    top_grids = grid_variances.head(50).index.tolist()

    print(f"\nAnalyzing top 50 grids by mismatch index variance...")

    for i, grid_id in enumerate(top_grids):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(top_grids)} grids...")

        grid_data = df[df['grid_id'] == grid_id].sort_values('tarih')

        if len(grid_data) < 30:
            continue

        signal = grid_data['mismatch_index'].fillna(method='ffill').values

        # PELT algorithm (Pruned Exact Linear Time)
        model = "l2"  # L2 cost (mean change detection)
        algo = rpt.Pelt(model=model, min_size=10, jump=1).fit(signal)

        # Detect breakpoints (penalty controls number of breaks)
        penalty = 3 * np.log(len(signal))  # BIC-like penalty
        breakpoints = algo.predict(pen=penalty)

        if len(breakpoints) > 1:  # Exclude endpoint
            # Map indices to dates
            dates = grid_data['tarih'].tolist()
            breakpoint_dates = [dates[bp-1] for bp in breakpoints[:-1]]

            changepoint_results.append({
                'grid_id': grid_id,
                'num_breakpoints': len(breakpoints) - 1,
                'breakpoint_dates': breakpoint_dates,
                'variance': grid_variances[grid_id]
            })

    print(f"\nChange-point detection complete:")
    print(f"  Grids with breakpoints: {len(changepoint_results)}")

    if changepoint_results:
        # Most frequent breakpoint dates (potential external events)
        all_dates = []
        for result in changepoint_results:
            all_dates.extend(result['breakpoint_dates'])

        date_counts = pd.Series(all_dates).value_counts()

        print(f"\nMost common breakpoint dates (potential system-wide events):")
        for date, count in date_counts.head(10).items():
            print(f"  {date.strftime('%Y-%m-%d')}: {count} grids")

        # Save
        df_cp = pd.DataFrame(changepoint_results)
        df_cp.to_csv('data/changepoint_analysis.csv', index=False)
        print("\n[SAVED] data/changepoint_analysis.csv")

except ImportError:
    print("\n[WARNING] ruptures not installed. Skipping change-point detection.")
    print("Install with: pip install ruptures")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nGenerated files:")
print("  1. data/weekday_weekend_comparison.csv - Hafta içi/sonu karşılaştırması")
print("  2. data/kpi_tail_analysis.csv - Basit KPI tail analizi")
print("  3. data/spatial_hotspot_analysis.csv - LISA hotspot kümeleri")
print("  4. data/changepoint_analysis.csv - Zaman serisi kırılma noktaları")

print("\nKey Findings:")
print(f"  - Grids with high weekday/weekend variance: {df_compare['high_weekend_variance'].sum() if 'df_compare' in locals() else 'N/A'}")
print(f"  - ML-KPI overlap (capacity util): {len(overlap) if 'overlap' in locals() else 'N/A'}/{len(ml_anomalies) if 'ml_anomalies' in locals() else 'N/A'}")
try:
    print(f"  - Undersupply hotspots (LISA HH): {len(hotspots)}")
    print(f"  - Oversupply coldspots (LISA LL): {len(coldspots)}")
except:
    pass
print(f"  - Grids with structural breaks: {len(changepoint_results) if 'changepoint_results' in locals() else 'N/A'}")
