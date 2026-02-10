"""
GÜNLÜK GRİD VERİSİ OLUŞTURMA - POSITION-BASED DISTRIBUTION

AMAÇ:
  Hat bazlı günlük verileri (ego_data_clean.csv) grid bazlı günlük verilere dönüştürmek.
  Bu veri, makine öğrenmesi anomaly detection ve regime shift analizleri için kullanılacak.

GİRDİ:
  - ego_data_clean.csv: Hat bazlı günlük yolcu/sefer/doluluk verileri (173 gün)
  - ego_route_stops_all_coords.csv: Hat duraklarının koordinatları

ÇIKTI:
  - daily_grid_data_position.csv: Grid-gün bazlı toplam veriler (POSITION-BASED)
    Format: Her satır = 1 grid × 1 gün
    Kolonlar: yolcu_sayisi, sefer_sayisi, doluluk_orani, vb.

DAĞITIM YÖNTEMİ:
  Position-Based (Terminal-Biased) Distribution:
  - İlk durak: 30% ağırlık
  - Son durak: 30% ağırlık
  - Orta duraklar: 40% / (N-2) ağırlık

  Transit literatüründe terminal durakların daha fazla biniş-iniş yaşadığı
  kabul edilir. Bu yöntem connectivity yerine durak pozisyonunu kullanır.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("CREATING DAILY GRID DATA - POSITION-BASED DISTRIBUTION")
print("=" * 70)

# ============================================================================
# ADIM 1: HAT BAZLI GÜNLÜK VERİLERİ YÜKLE
# ============================================================================
print("\n1. Loading daily route data...")
df_daily = pd.read_csv('data/ego_data_with_dates_CORRECTED.csv', encoding='utf-8-sig')
df_daily['TARIH'] = pd.to_datetime(df_daily['TARIH'])

# Özet satırları temizle
df_daily = df_daily[df_daily['HAT NO'].notna()].copy()
df_daily = df_daily[df_daily['HAT NO'] != ''].copy()
df_daily['HAT NO'] = df_daily['HAT NO'].astype(str).str.strip()

print(f"   Total route-day records: {len(df_daily):,}")
print(f"   Date range: {df_daily['TARIH'].min()} to {df_daily['TARIH'].max()}")
print(f"   Unique dates: {df_daily['TARIH'].nunique()}")
print(f"   Unique routes: {df_daily['HAT NO'].nunique()}")

# ============================================================================
# ADIM 2: DURAK KOORDİNATLARINI YÜKLE
# ============================================================================
print("\n2. Loading stop coordinates...")
df_coords = pd.read_csv('data/ego_route_stops_all_coords.csv', encoding='utf-8-sig')

# Ankara sınırlarına filtrele
ANKARA_BOUNDS = {'lat': (39.5, 40.3), 'lon': (32.3, 33.2)}
df_coords = df_coords[
    (df_coords['final_latitude'] >= ANKARA_BOUNDS['lat'][0]) &
    (df_coords['final_latitude'] <= ANKARA_BOUNDS['lat'][1]) &
    (df_coords['final_longitude'] >= ANKARA_BOUNDS['lon'][0]) &
    (df_coords['final_longitude'] <= ANKARA_BOUNDS['lon'][1])
].copy()

df_coords['hat_adi'] = df_coords['hat_adi'].astype(str).str.strip()
print(f"   Stops in Ankara: {len(df_coords):,}")
print(f"   Unique routes in coords: {df_coords['hat_adi'].nunique()}")

# ============================================================================
# ADIM 3: GÜNLÜK VERİLERİ KOORDİNATLARLA BİRLEŞTİR
# ============================================================================
print("\n3. Merging daily data with coordinates...")
df_merged = df_coords.merge(
    df_daily,
    left_on='hat_adi',
    right_on='HAT NO',
    how='inner'
)

print(f"   Merged records: {len(df_merged):,}")
print(f"   Dates after merge: {df_merged['TARIH'].nunique()}")

# ============================================================================
# POSITION-BASED DISTRIBUTION (Terminal-Biased)
# ============================================================================
# MANTIK: Transit literatüründe terminal duraklar daha fazla yolcu çeker
#         - İlk durak: biniş bölgesi (30%)
#         - Son durak: iniş bölgesi (30%)
#         - Orta duraklar: transfer bölgesi (40% paylaşımlı)
#
# FORMÜL:
#   İlk durak (sira=1): W = 0.30
#   Son durak (sira=max): W = 0.30
#   Orta duraklar: W = 0.40 / (N-2)
#
# ÖRNEK:
#   Hat 101: 10,000 yolcu, 50 durak
#   - İlk durak: 10,000 × 0.30 = 3,000 yolcu
#   - Son durak: 10,000 × 0.30 = 3,000 yolcu
#   - Her orta durak: 10,000 × (0.40/48) = 83 yolcu
print("\n3b. Applying position-based distribution (terminal-biased)...")

# ADIM 1: Her hat için maksimum sırayı hesapla
print("   Step 1: Calculating max stop position per route...")
df_merged['max_sira'] = df_merged.groupby(['HAT NO', 'TARIH'])['sira'].transform('max')

# ADIM 2: Durak pozisyonunu belirle (first, last, middle)
df_merged['stop_position'] = 'middle'
df_merged.loc[df_merged['sira'] == 1, 'stop_position'] = 'first'
df_merged.loc[df_merged['sira'] == df_merged['max_sira'], 'stop_position'] = 'last'

# İstatistikler
position_counts = df_merged['stop_position'].value_counts()
print(f"     First stops: {position_counts.get('first', 0):,}")
print(f"     Last stops: {position_counts.get('last', 0):,}")
print(f"     Middle stops: {position_counts.get('middle', 0):,}")

# ADIM 3: Her hat için orta durak sayısını hesapla
df_merged['middle_stop_count'] = (df_merged['max_sira'] - 2).clip(lower=1)

# ADIM 4: Position-based ağırlıkları ata
df_merged['stop_weight'] = 0.0

# Terminal duraklar: %30 her biri
df_merged.loc[df_merged['stop_position'] == 'first', 'stop_weight'] = 0.30
df_merged.loc[df_merged['stop_position'] == 'last', 'stop_weight'] = 0.30

# Orta duraklar: %40 / (N-2)
df_merged.loc[df_merged['stop_position'] == 'middle', 'stop_weight'] = (
    0.40 / df_merged.loc[df_merged['stop_position'] == 'middle', 'middle_stop_count']
)

# ADIM 5: Hat-gün toplamlarını ağırlıklara göre dağıt
print("   Step 2: Distributing passengers by position weight...")
df_merged['yolcu_per_stop'] = df_merged['TAŞINAN YOLCU SAYISI'] * df_merged['stop_weight']
df_merged['sefer_per_stop'] = df_merged['SEFER SAYISI'] * df_merged['stop_weight']
df_merged['kapasite_per_stop'] = df_merged['ARAÇ KAPASİTESİ'] * df_merged['stop_weight']

print(f"   Position-based distribution applied:")
print(f"     Avg passengers per stop: {df_merged['yolcu_per_stop'].mean():.1f}")
print(f"     Avg trips per stop: {df_merged['sefer_per_stop'].mean():.1f}")
print(f"     Avg capacity per stop: {df_merged['kapasite_per_stop'].mean():.1f}")
print(f"     Min stop weight: {df_merged['stop_weight'].min():.4f}")
print(f"     Max stop weight: {df_merged['stop_weight'].max():.4f}")

# Ağırlık dağılımı istatistikleri
print(f"\n   Weight distribution by position:")
for position in ['first', 'last', 'middle']:
    mask = df_merged['stop_position'] == position
    if mask.sum() > 0:
        avg_weight = df_merged.loc[mask, 'stop_weight'].mean()
        print(f"     {position.capitalize()}: avg weight = {avg_weight:.4f}")

# ============================================================================
# ADIM 4: GRİD HÜCRELERİ OLUŞTUR (1km × 1km)
# ============================================================================
print("\n4. Creating 1km grid cells...")
GRID_SIZE_METERS = 1000
METERS_PER_DEGREE_LAT = 111000
METERS_PER_DEGREE_LON = 85000

GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON

df_merged['grid_lat'] = np.floor(df_merged['final_latitude'] / GRID_SIZE_LAT) * GRID_SIZE_LAT
df_merged['grid_lon'] = np.floor(df_merged['final_longitude'] / GRID_SIZE_LON) * GRID_SIZE_LON

# ============================================================================
# ADIM 5: TARİH + GRİD BAZINDA TOPLA (AGGREGATION)
# ============================================================================
print("\n5. Aggregating by date and grid cell (using per-stop metrics)...")
df_grid_daily = df_merged.groupby(['TARIH', 'grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',
    'yolcu_per_stop': 'mean',
    'sefer_per_stop': 'mean',
    'kapasite_per_stop': 'mean',
    'DOLULUK ORANI': 'mean'
}).reset_index()

df_grid_daily.columns = ['tarih', 'grid_lat', 'grid_lon', 'stop_count',
                         'yolcu_per_stop', 'sefer_per_stop', 'kapasite_per_stop', 'doluluk_orani']

# Grid ID oluştur
df_grid_daily['grid_id'] = (
    df_grid_daily['grid_lat'].round(4).astype(str) + '_' +
    df_grid_daily['grid_lon'].round(4).astype(str)
)

print(f"   Total grid-day records: {len(df_grid_daily):,}")
print(f"   Unique grid cells: {df_grid_daily['grid_id'].nunique()}")
print(f"   Unique dates: {df_grid_daily['tarih'].nunique()}")
print(f"   Average grids per day: {len(df_grid_daily) / df_grid_daily['tarih'].nunique():.0f}")

# ============================================================================
# ADIM 6: EK ÖZELLİKLER HESAPLA
# ============================================================================
print("\n6. Calculating additional metrics...")

df_grid_daily['kapasite_kullanimi'] = (
    df_grid_daily['yolcu_per_stop'] / df_grid_daily['kapasite_per_stop'].replace(0, 1)
) * 100

# ============================================================================
# ADIM 7: YETERLİ VERİYE SAHİP GRİD'LERİ SÜZME
# ============================================================================
print("\n7. Filtering grids with sufficient data...")
grid_day_counts = df_grid_daily.groupby('grid_id')['tarih'].nunique()
valid_grids = grid_day_counts[grid_day_counts >= 30].index.tolist()

df_grid_daily_filtered = df_grid_daily[df_grid_daily['grid_id'].isin(valid_grids)].copy()

print(f"   Grids with >=30 days: {len(valid_grids)}")
print(f"   Filtered records: {len(df_grid_daily_filtered):,}")

# ============================================================================
# ADIM 8: KAYDET
# ============================================================================
print("\n8. Saving daily grid data (position-based)...")
output_file = 'data/daily_grid_data_position.csv'
df_grid_daily_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   [SAVED] {output_file}")

# ============================================================================
# ADIM 9: ÖZET İSTATİSTİKLER
# ============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY STATISTICS - POSITION-BASED DISTRIBUTION")
print("=" * 70)
print(f"\nDate range: {df_grid_daily_filtered['tarih'].min()} to {df_grid_daily_filtered['tarih'].max()}")
print(f"Total days: {df_grid_daily_filtered['tarih'].nunique()}")
print(f"Total grid cells: {df_grid_daily_filtered['grid_id'].nunique()}")
print(f"Total grid-day records: {len(df_grid_daily_filtered):,}")

print(f"\nMetrics per grid-day:")
print(f"  Avg passengers per stop: {df_grid_daily_filtered['yolcu_per_stop'].mean():,.1f}")
print(f"  Avg trips per stop: {df_grid_daily_filtered['sefer_per_stop'].mean():.1f}")
print(f"  Avg capacity per stop: {df_grid_daily_filtered['kapasite_per_stop'].mean():,.1f}")
print(f"  Avg occupancy: {df_grid_daily_filtered['doluluk_orani'].mean():.1f}%")
print(f"  Avg capacity utilization: {df_grid_daily_filtered['kapasite_kullanimi'].mean():.1f}%")
print(f"  Avg stops per grid: {df_grid_daily_filtered['stop_count'].mean():.1f}")

print("\nDistribution method: Position-Based (Terminal-Biased)")
print("  - First stops: 30% weight")
print("  - Last stops: 30% weight")
print("  - Middle stops: 40% / (N-2) weight")

print("\nData ready for comparison with connectivity-based method!")
