"""
GÜNLÜK GRİD VERİSİ OLUŞTURMA

AMAÇ:
  Hat bazlı günlük verileri (ego_data_clean.csv) grid bazlı günlük verilere dönüştürmek.
  Bu veri, makine öğrenmesi anomaly detection ve regime shift analizleri için kullanılacak.

GİRDİ:
  - ego_data_clean.csv: Hat bazlı günlük yolcu/sefer/doluluk verileri (173 gün)
  - ego_route_stops_all_coords.csv: Hat duraklarının koordinatları

ÇIKTI:
  - daily_grid_data.csv: Grid-gün bazlı toplam veriler
    Format: Her satır = 1 grid × 1 gün
    Kolonlar: yolcu_sayisi, sefer_sayisi, doluluk_orani, vb.

NEDEN BU DÖNÜŞÜM GEREKLİ?
  1. Hatlar sabit değil, grid'ler sabit → temporal analiz için grid bazlı veri şart
  2. Rejim kayması analizi için aynı grid'in günler boyunca nasıl değiştiğini görmek gerekiyor
  3. Anomaly detection için grid-level ortalamalar ve özellikler gerekli

METODOLOJİ:
  1. Hat verilerini durak koordinatlarıyla birleştir
  2. Her durak koordinatını 1km×1km grid hücresine ata (lat-lon snapping)
  3. Her tarih + grid için: yolcu, sefer, doluluk vb. topla/ortala
  4. En az 30 günlük verisi olan grid'leri tut (temporal analysis için yeterli data)
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("CREATING DAILY GRID DATA")
print("=" * 70)

# ============================================================================
# ADIM 1: HAT BAZLI GÜNLÜK VERİLERİ YÜKLE
# ============================================================================
# ego_data_clean.csv: PDF'lerden parse edilen hat bazlı günlük veriler
# Her satır = 1 hat × 1 gün (örn: Hat 101, 2023-12-25)
# Kolonlar: TARIH, HAT NO, SEFER SAYISI, TAŞINAN YOLCU SAYISI, DOLULUK ORANI vb.
print("\n1. Loading daily route data...")
df_daily = pd.read_csv('data/ego_data_with_dates_CORRECTED.csv', encoding='utf-8-sig')
df_daily['TARIH'] = pd.to_datetime(df_daily['TARIH'])

# Özet satırları temizle (HAT NO boş olanlar toplam satırları)
df_daily = df_daily[df_daily['HAT NO'].notna()].copy()
df_daily = df_daily[df_daily['HAT NO'] != ''].copy()
df_daily['HAT NO'] = df_daily['HAT NO'].astype(str).str.strip()

print(f"   Total route-day records: {len(df_daily):,}")
print(f"   Date range: {df_daily['TARIH'].min()} to {df_daily['TARIH'].max()}")
print(f"   Unique dates: {df_daily['TARIH'].nunique()}")
print(f"   Unique routes: {df_daily['HAT NO'].nunique()}")

# NOTE: Using ego_data_clean.csv (already deduplicated by create_clean_deduplicated_data.py)
# No need to deduplicate here - data is already clean!

# ============================================================================
# ADIM 2: DURAK KOORDİNATLARINI YÜKLE
# ============================================================================
# ego_route_stops_all_coords.csv: Her hat için durakların GPS koordinatları
# Geoapify/Google Maps ile geocode edilmiş
# Her satır = 1 hat × 1 durak (örn: Hat 101, Kızılay Durağı, 39.9°N 32.8°E)
print("\n2. Loading stop coordinates...")
df_coords = pd.read_csv('data/ego_route_stops_all_coords.csv', encoding='utf-8-sig')

# Ankara sınırlarına filtrele (outlier koordinatları çıkar)
# Ankara merkez: ~39.9°N, 32.8°E
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
# Her hat için: o hat'ın günlük yolcu/sefer verilerini, o hat'ın duraklarıyla eşleştir
# Sonuç: Her satır = 1 durak × 1 gün × 1 hat
# Örnek: Kızılay durağı (39.9, 32.8), Hat 101, 2023-12-25, 500 yolcu
print("\n3. Merging daily data with coordinates...")
df_merged = df_coords.merge(
    df_daily,
    left_on='hat_adi',       # Koordinat dosyasındaki hat adı
    right_on='HAT NO',       # Günlük veri dosyasındaki hat numarası
    how='inner'              # Sadece eşleşenleri tut
)

print(f"   Merged records: {len(df_merged):,}")
print(f"   Dates after merge: {df_merged['TARIH'].nunique()}")

# ============================================================================
# FIX: AĞIRLIKLI DAĞITIM (Weighted Distribution by Stop Connectivity)
# ============================================================================
# PROBLEM: Uniform dağıtım (1/N) şehri düz bir kağıt gibi görür.
#          Oysa gerçekte hub duraklar (Kızılay, Ulus) daha fazla yolcu üretir.
#
# ÇÖZÜM: Transfer gücüne dayalı ağırlıklı dağıtım:
#        - Bir duraktan kaç hat geçiyorsa, o kadar yüksek ağırlık
#        - Hub duraklar (50 hat) >>> İzole duraklar (1 hat)
#
# FORMÜL:
#   W_durak = (Duraktan Geçen Hat Sayısı) / (Hattın Tüm Duraklarının Toplam Bağlantısı)
#   yolcu_durak = Toplam_Yolcu × W_durak
#
# ÖRNEK:
#   Hat 101: 10,000 yolcu, 50 durak
#   - Kızılay durağı: 50 hat geçiyor → W = 50/500 = 0.10 → 1000 yolcu ✅
#   - İzole durak: 1 hat geçiyor → W = 1/500 = 0.002 → 20 yolcu ✅
print("\n3b. Applying weighted distribution based on stop connectivity...")

# ADIM 1: Her durağın bağlantı derecesini hesapla (kaç farklı hat geçiyor)
print("   Step 1: Calculating stop connectivity (transfer power)...")
stop_connectivity = df_coords.groupby('durak_kodu')['hat_adi'].nunique().reset_index()
stop_connectivity.columns = ['durak_kodu', 'hat_sayisi']

print(f"     Stops with connectivity data: {len(stop_connectivity):,}")
print(f"     Max routes through single stop: {stop_connectivity['hat_sayisi'].max()}")
print(f"     Avg routes per stop: {stop_connectivity['hat_sayisi'].mean():.1f}")

# ADIM 2: Merge ile her stop'a bağlantı derecesini ekle
df_merged = df_merged.merge(stop_connectivity, on='durak_kodu', how='left')
df_merged['hat_sayisi'] = df_merged['hat_sayisi'].fillna(1)  # Eğer veri yoksa, en az 1 hat

# ADIM 3: Her hat-gün için toplam bağlantı skorunu hesapla
print("   Step 2: Computing route-level total connectivity...")
df_merged['route_total_connectivity'] = df_merged.groupby(['HAT NO', 'TARIH'])['hat_sayisi'].transform('sum')

# ADIM 4: Ağırlıklı dağıtım - her durağın ağırlığı = durak bağlantısı / hat toplam bağlantısı
print("   Step 3: Distributing passengers by connectivity weight...")
df_merged['stop_weight'] = df_merged['hat_sayisi'] / df_merged['route_total_connectivity']

# ADIM 5: Hat-gün toplamlarını ağırlıklara göre dağıt
df_merged['yolcu_per_stop'] = df_merged['TAŞINAN YOLCU SAYISI'] * df_merged['stop_weight']
df_merged['sefer_per_stop'] = df_merged['SEFER SAYISI'] * df_merged['stop_weight']
df_merged['kapasite_per_stop'] = df_merged['ARAÇ KAPASİTESİ'] * df_merged['stop_weight']

print(f"   Weighted distribution applied:")
print(f"     Avg passengers per stop: {df_merged['yolcu_per_stop'].mean():.1f}")
print(f"     Avg trips per stop: {df_merged['sefer_per_stop'].mean():.1f}")
print(f"     Avg capacity per stop: {df_merged['kapasite_per_stop'].mean():.1f}")
print(f"     Min stop weight: {df_merged['stop_weight'].min():.4f}")
print(f"     Max stop weight: {df_merged['stop_weight'].max():.4f}")

# ============================================================================
# ADIM 4: GRİD HÜCRELERİ OLUŞTUR (1km × 1km)
# ============================================================================
# Her koordinatı en yakın grid hücresine "snap" et
# Grid boyutu: 1km × 1km (analizler için uygun granülarite)
#
# NEDEN 1km?
#   - Çok küçük (100m) → fazla grid, sparse data
#   - Çok büyük (5km) → detay kaybı
#   - 1km → dengeli (ortalama mahalle boyutu)
#
# MATEMATİK:
#   - Latitude: 1° ≈ 111km → 1km ≈ 0.009°
#   - Longitude: 1° ≈ 85km (Ankara enleminde) → 1km ≈ 0.012°
#   - Floor function: 39.923° → 39.920° (en yakın grid köşesi)
print("\n4. Creating 1km grid cells...")
GRID_SIZE_METERS = 1000
METERS_PER_DEGREE_LAT = 111000  # 1 derece latitude ≈ 111km
METERS_PER_DEGREE_LON = 85000   # 1 derece longitude ≈ 85km (Ankara'da)

GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT  # ≈0.009°
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON  # ≈0.012°

# Her koordinatı grid'e snap et (floor function)
df_merged['grid_lat'] = np.floor(df_merged['final_latitude'] / GRID_SIZE_LAT) * GRID_SIZE_LAT
df_merged['grid_lon'] = np.floor(df_merged['final_longitude'] / GRID_SIZE_LON) * GRID_SIZE_LON

# ============================================================================
# ADIM 5: TARİH + GRİD BAZINDA TOPLA (AGGREGATION)
# ============================================================================
# Her gün, her grid için: o grid'deki duraklardan geçen hatların normalized metriklerini hesapla
#
# YENİ AGGREGATION MANTIĞI (Duplikasyon Fix):
#   - stop_count: O grid'de kaç durak var? (count)
#   - yolcu_per_stop: Ortalama durak başına yolcu (mean - normalized metrik)
#   - sefer_per_stop: Ortalama durak başına sefer (mean - normalized metrik)
#   - kapasite_per_stop: Ortalama durak başına kapasite (mean - normalized metrik)
#   - doluluk_orani: Ortalama doluluk oranı (mean)
#
# FARK:
#   Eski: sum(toplam_yolcu) → duplikasyon nedeniyle şişirilmiş değer
#   Yeni: mean(yolcu_per_stop) → gerçek normalized değer
#
# SONUÇ: Her satır = 1 tarih × 1 grid
# Örnek: 2023-12-25, Grid (39.92, 32.86), 5 durak, ort. 240 yolcu/durak
print("\n5. Aggregating by date and grid cell (using per-stop metrics)...")
df_grid_daily = df_merged.groupby(['TARIH', 'grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',          # Durak sayısı (o grid'de kaç durak var?)
    'yolcu_per_stop': 'mean',       # Ortalama durak başına yolcu
    'sefer_per_stop': 'mean',       # Ortalama durak başına sefer
    'kapasite_per_stop': 'mean',    # Ortalama durak başına kapasite
    'DOLULUK ORANI': 'mean'         # Ortalama doluluk oranı
}).reset_index()

df_grid_daily.columns = ['tarih', 'grid_lat', 'grid_lon', 'stop_count',
                         'yolcu_per_stop', 'sefer_per_stop', 'kapasite_per_stop', 'doluluk_orani']

# Grid ID oluştur (kolay referans için)
# Format: "39.9279_32.8588" (lat_lon)
df_grid_daily['grid_id'] = (
    df_grid_daily['grid_lat'].round(4).astype(str) + '_' +
    df_grid_daily['grid_lon'].round(4).astype(str)
)

print(f"   Total grid-day records: {len(df_grid_daily):,}")
print(f"   Unique grid cells: {df_grid_daily['grid_id'].nunique()}")
print(f"   Unique dates: {df_grid_daily['tarih'].nunique()}")
print(f"   Average grids per day: {len(df_grid_daily) / df_grid_daily['tarih'].nunique():.0f}")

# ============================================================================
# ADIM 6: EK ÖZELLİKLER HESAPLA (FEATURE ENGINEERING)
# ============================================================================
# Anomaly detection ve ML modelleri için ek metrikler üret
# NOT: Artık yolcu_per_stop, sefer_per_stop, kapasite_per_stop zaten var (Adım 5'te hesaplandı)
# Sadece kapasite kullanım oranını hesaplamak gerekiyor
print("\n6. Calculating additional metrics...")

# Kapasite kullanım oranı: Talep/Hizmet dengesi (per-stop bazında)
#    %100'ün üstü → talep > hizmet (yetersiz hizmet)
#    %100'ün altı → talep < hizmet (fazla hizmet veya düşük talep)
df_grid_daily['kapasite_kullanimi'] = (
    df_grid_daily['yolcu_per_stop'] / df_grid_daily['kapasite_per_stop'].replace(0, 1)
) * 100

# ============================================================================
# ADIM 7: YETERLİ VERİYE SAHİP GRİD'LERİ SÜZME
# ============================================================================
# Temporal analiz (rejim kayması) için her grid'in yeterli gün sayısına ihtiyacı var
#
# NEDEN EN AZ 30 GÜN?
#   - Rejim kayması analizi: günler arası değişimi ölçer
#   - Embedding stability: ardışık günler arası mesafe hesaplar
#   - Clustering: her gün için cluster assignment gerekir
#   - <30 gün veri → güvenilir temporal pattern yakalanamaz
#
# ÖRNEK:
#   - Grid A: 150 günlük veri → tutulur (temporal pattern net)
#   - Grid B: 15 günlük veri → atılır (çok az veri, sparse)
print("\n7. Filtering grids with sufficient data...")
grid_day_counts = df_grid_daily.groupby('grid_id')['tarih'].nunique()
valid_grids = grid_day_counts[grid_day_counts >= 30].index.tolist()

df_grid_daily_filtered = df_grid_daily[df_grid_daily['grid_id'].isin(valid_grids)].copy()

print(f"   Grids with >=30 days: {len(valid_grids)}")
print(f"   Filtered records: {len(df_grid_daily_filtered):,}")

# 8. Save
print("\n8. Saving daily grid data...")
output_file = 'data/daily_grid_data.csv'
df_grid_daily_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   [SAVED] {output_file}")

# 9. Summary stats
print(f"\n{'=' * 70}")
print("SUMMARY STATISTICS")
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

print("\nData ready for:")
print("  1. Autoencoder anomaly detection")
print("  2. Isolation Forest / LOF")
print("  3. Embedding stability analysis")
print("  4. Daily clustering regime shift analysis")
