"""
GRİD ANOMALY DETECTION & REGIME SHIFT ANALYSIS
Makine Semiyotiği Perspektifiyle Hizmet-Talep Uyuşmazlığı ve Rejim Kayması Analizi

====================================================================================================
GENEL BAKIŞ
====================================================================================================

PROBLEM:
  Bazı grid'lerde hizmet-talep dengesi bozuk olabilir VEYA grid'ler temporal olarak
  istikrarsız davranış sergileyebilir. Bu problemler için:
    ❌ "Doğru hizmet" etiketi yok → Supervised learning kullanılamaz
    ❌ "İdeal kapasite" formülü yok → Kural bazlı sistem yetersiz
    ✅ Sadece sistemin kendi normunu gözlemleyebiliriz → UNSUPERVISED LEARNING

TEMEL SORULAR:
  1. Hangi grid'ler sistemin "normal" kabul ettiği talep-hizmet ilişkisine uymuyorlar?
     → Anomaly Detection (4 algoritma ile consensus)

  2. Hangi grid'ler günler boyunca sistemin temsil uzayında stabil değiller?
     → Regime Shift Analysis (2 yöntemle tespit)

  3. Hangi grid'ler hem anomalous hem de unstable?
     → Critical Grids (öncelikli müdahale gerektirir)

====================================================================================================
BÖLÜM 1: HİZMET-TALEP UYUŞMAZLIĞI (UNSUPERVISED ANOMALY DETECTION)
====================================================================================================

MANTIK:
  "Benzer talep davranışına sahip grid'lerin hizmet açısından nerede ayrıştığını bul"

  Sistem kendi "normal" talep-hizmet ilişkisini öğrenir. Bu ilişkiye uymayan grid'ler
  anomalidir. Bu gridlerde:
    - Yüksek talep + düşük hizmet → yetersiz hizmet
    - Düşük talep + yüksek hizmet → fazla hizmet (over-provisioning)
    - İlişkisi düşük boyutlu pattern'e sığmayan → sistem tarafından anlaşılamayan

ALGORİTMALAR:
  1. AUTOENCODER (PCA-based):
     - 7 boyutlu veriyi 3 boyuta sıkıştır, sonra tekrar 7'ye aç
     - Yüksek reconstruction error = düşük boyuta sığmayan grid = anomali
     - Mantık: Normal gridler 3 temel pattern'le açıklanır, anomaliler açıklanamaz

  2. ISOLATION FOREST:
     - 100 decision tree oluştur, her grid'i izole etmeye çalış
     - Kısa path length = kolay izole = anomali
     - Mantık: Anomaliler feature space'te "uzak köşelerde" durur

  3. LOCAL OUTLIER FACTOR (LOF):
     - Her grid için 20 en yakın komşuyu bul
     - Komşularına göre yoğunluk farkı = anomaly score
     - Mantık: Yoğun bölgede yalnız grid = anomali

  4. GRAPH-BASED SPATIAL:
     - Mekânsal komşuluk grafiği oluştur (1.5km threshold)
     - Komşularından farklı olan grid = anomali
     - Mantık: Yan yana gridler benzer olmalı (mekânsal süreklilik)

CONSENSUS:
  4 yöntemden EN AZ 2'si onaylamalı → robust tespit

====================================================================================================
BÖLÜM 2: REJİM KAYMASI (TEMPORAL STABILITY ANALYSIS)
====================================================================================================

MANTIK:
  "Aynı grid'in günlük temsili stabil mi yoksa sürekli değişiyor mu?"

  Eğer bir grid her gün farklı bir "temsil noktası"na düşüyorsa:
    → Sistem o grid'i tutarlı şekilde kategorize edemiyor
    → Epistemolojik belirsizlik (makine semiyotiği eşik problemi)

YÖNTEMLER:
  1. EMBEDDING STABILITY:
     - Her gün için tüm grid'leri 2D PCA embedding'e dönüştür
     - Her grid için ardışık günler arası Euclidean mesafe hesapla
     - Yüksek ortalama mesafe = istikrarsız grid
     - Mantık: Stabil grid embedding uzayında aynı yerde kalır

  2. DAILY CLUSTERING REGIME SHIFT:
     - Her gün K-Means ile 5 cluster oluştur
     - Her grid için cluster değişim oranını hesapla
     - Yüksek switch rate = istikrarsız grid
     - Mantık: Sistem grid'i tutarlı bir kategoriye atayamıyor

SONUÇ:
  En az birinde instability varsa → rejim kayması var
  Her ikisinde de varsa → çok güçlü rejim kayması

====================================================================================================
MAKİNE SEMİYOTİĞİ YORUMU
====================================================================================================

"Sistem aynı mekânı her gün farklı bir kategoriye itiyor"

Normal grid:
  Embedding uzayı:  ●───●───●───●  (stabil trajektori)
  Cluster labels:   [C1─C1─C1─C1]  (aynı cluster)
  → Sistem bu grid'i tutarlı şekilde okuyor

Kritik grid (anomali + rejim kayması):
  Embedding uzayı:  ●     ●   ●  ● (kaotik trajektori)
  Cluster labels:   [C2─C1─C3─C4]  (sürekli değişiyor)
  → Sistem bu grid'i kategorize edemiyor → EŞİK PROBLEMİ

Grid'in fiziksel konumu sabit ama sistemin temsilsel okuyuşu tutarsız!
Bu, talep-hizmet dengesinin sistemin epistemolojik çerçevesine sığmadığını gösterir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import zscore
from scipy.optimize import linear_sum_assignment
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GRID ANOMALY & REGIME SHIFT ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 0: LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("PART 0: LOADING DATA")
print("=" * 80)

df = pd.read_csv('data/daily_grid_data.csv', encoding='utf-8-sig')
df['tarih'] = pd.to_datetime(df['tarih'])

print(f"\nTotal records: {len(df):,}")
print(f"Date range: {df['tarih'].min()} to {df['tarih'].max()}")
print(f"Unique grids: {df['grid_id'].nunique()}")
print(f"Unique dates: {df['tarih'].nunique()}")

# Feature engineering: Grid-level aggregated features (average across all days)
print("\nAggregating grid-level features (for anomaly detection)...")
df_grid_agg = df.groupby('grid_id').agg({
    'grid_lat': 'first',
    'grid_lon': 'first',
    'stop_count': 'mean',
    'yolcu_per_stop': 'mean',      # Ortalama durak başına yolcu
    'sefer_per_stop': 'mean',      # Ortalama durak başına sefer
    'kapasite_per_stop': 'mean',   # Ortalama durak başına kapasite
    'doluluk_orani': 'mean',
    'kapasite_kullanimi': 'mean'
}).reset_index()

print(f"Grid aggregated features: {df_grid_agg.shape}")

# ============================================================================
# PART 1: SERVICE-DEMAND MISMATCH (ANOMALY DETECTION)
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: SERVICE-DEMAND MISMATCH ANALYSIS")
print("=" * 80)

# ============================================================================
# ADIL KARŞILAŞTIRMA PARAMETRELERİ
# ============================================================================
# Tüm algoritmalar aynı anomaly rate'i hedefleyecek (percentile-based threshold)
# Bu sayede "119 vs 48" gibi farklar algoritma kalitesinden kaynaklanır, ayardan değil

TARGET_ANOMALY_RATE = 0.05  # %5 - En kritik gridler
PERCENTILE_THRESHOLD = (1 - TARGET_ANOMALY_RATE) * 100  # 95th percentile

print(f"\n*** ADIL KIYASLAMA MODU AKTIF ***")
print(f"    Target Anomaly Rate: {TARGET_ANOMALY_RATE:.1%}")
print(f"    Percentile Threshold: {PERCENTILE_THRESHOLD:.1f}th")
print(f"    Hedef tespit sayisi: ~{int(len(df_grid_agg) * TARGET_ANOMALY_RATE)} grid")

# Select features for anomaly detection
# Focus on: demand (yolcu_per_stop), service (sefer_per_stop, kapasite_per_stop),
#           and mismatch (doluluk_orani, kapasite_kullanimi)
# NOTE: Using normalized per-stop metrics to avoid duplication bias
feature_cols = [
    'stop_count',           # Grid'deki durak yoğunluğu
    'yolcu_per_stop',       # Durak başına ortalama yolcu (normalized demand)
    'sefer_per_stop',       # Durak başına ortalama sefer (normalized service)
    'kapasite_per_stop',    # Durak başına ortalama kapasite (normalized capacity)
    'doluluk_orani',        # Ortalama doluluk oranı
    'kapasite_kullanimi'    # Kapasite kullanım oranı (demand/supply balance)
]

print(f"\nFeatures for anomaly detection: {feature_cols}")

X = df_grid_agg[feature_cols].values
print(f"Feature matrix shape: {X.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures for anomaly detection: {feature_cols}")
print(f"Feature matrix shape: {X_scaled.shape}")

# ============================================================================
# 1.1 AUTOENCODER (PCA-based Reconstruction Error)
# ============================================================================
# MANTIK: Sistemin "normal" kabul ettiği talep-hizmet ilişkisini öğren.
#         Bu ilişkiye sığmayan gridler anomalidir.
#
# NASIL ÇALIŞIR?
#   1. Veriyi düşük boyuta sıkıştır (7D → 3D)
#   2. Tekrar yüksek boyuta aç (3D → 7D)
#   3. Orijinal ile yeniden oluşturulmuş arasındaki fark = reconstruction error
#   4. Yüksek error = düşük boyuta sığmayan grid = ANOMALI
#
# NEDEN PCA?
#   - PCA varyansın çoğunu (≈%91) 3 bileşende tutar
#   - Bu 3 bileşen "normal" gridleri iyi temsil eder
#   - Anomali gridler bu 3 pattern'e sığmaz → error↑
#
# MATEMATİK:
#   X_compressed = PCA.transform(X)    # 7D → 3D (encoder)
#   X_reconstructed = PCA.inverse(X_compressed)  # 3D → 7D (decoder)
#   error_i = mean((X[i] - X_reconstructed[i])²)
# ============================================================================

print("\n" + "-" * 80)
print("1.1 AUTOENCODER (PCA-based Reconstruction Error)")
print("-" * 80)

# PCA ile 3 principal component kullan
# NEDEN 3? → Varyansın %90+ açıklıyor, daha az → bilgi kaybı, daha fazla → noise
n_components = 3
pca = PCA(n_components=n_components)

# ENCODER: 7 boyut → 3 boyut (sıkıştırma)
X_compressed = pca.fit_transform(X_scaled)

# DECODER: 3 boyut → 7 boyut (yeniden oluşturma)
X_reconstructed = pca.inverse_transform(X_compressed)

# Reconstruction error hesapla (her grid için)
# Mean Squared Error (MSE) kullanıyoruz
reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

print(f"\nPCA components: {n_components}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Mean reconstruction error: {reconstruction_errors.mean():.4f}")
print(f"Std reconstruction error: {reconstruction_errors.std():.4f}")

# ADIL KIYASLAMA: Percentile-based threshold (mean+2std yerine)
threshold_ae = np.percentile(reconstruction_errors, PERCENTILE_THRESHOLD)
anomalies_ae = reconstruction_errors > threshold_ae

df_grid_agg['reconstruction_error'] = reconstruction_errors
df_grid_agg['anomaly_autoencoder'] = anomalies_ae

print(f"Anomaly threshold (percentile): {threshold_ae:.4f}")
print(f"Anomalies detected: {anomalies_ae.sum()} ({100*anomalies_ae.sum()/len(df_grid_agg):.1f}%)")

# ============================================================================
# 1.2 ISOLATION FOREST
# ============================================================================
# MANTIK: Anomaliler izole edilmesi kolay olan noktalardır
#
# NASIL ÇALIŞIR?
#   1. 100 decision tree oluştur
#   2. Her tree'de rastgele özellik + rastgele değer ile split yap
#   3. Her grid için path length hesapla (kaç split'te izole oldu?)
#   4. Kısa path = kolay izole = ANOMALI
#
# NEDEN İŞLER?
#   - Normal gridler kümeler halinde → izole etmek için çok split gerekir
#   - Anomali gridler yalnız → 2-3 split ile izole olur
#
# ÖRNEK:
#   Normal grid:  Root → Split1 → Split2 → ... → Split10 (path_length=10)
#   Anomaly grid: Root → Split1 → ISOLATED (path_length=2)
#
# CONTAMINATION PARAMETER:
#   - contamination=0.1 → verinin %10'unun anomali olmasını bekliyoruz
#   - Bu threshold'u belirler
# ============================================================================

print("\n" + "-" * 80)
print("1.2 ISOLATION FOREST")
print("-" * 80)

iso_forest = IsolationForest(
    contamination=TARGET_ANOMALY_RATE,  # ADIL KIYASLAMA: Aynı oran hedefleniyor
    random_state=42,    # Reproducibility için seed
    n_estimators=100    # 100 decision tree (ensemble)
)
print(f"Contamination: {TARGET_ANOMALY_RATE:.1%} (adil kıyas)")

# Model fit + predict (tek adımda)
predictions_if = iso_forest.fit_predict(X_scaled)

# Anomaly scores (daha negatif = daha anomalous)
anomaly_scores_if = iso_forest.score_samples(X_scaled)

# -1 = anomaly, 1 = normal
anomalies_if = predictions_if == -1

df_grid_agg['anomaly_score_if'] = anomaly_scores_if
df_grid_agg['anomaly_isolation_forest'] = anomalies_if

print(f"Contamination: {TARGET_ANOMALY_RATE:.1%}")
print(f"Anomalies detected: {anomalies_if.sum()} ({100*anomalies_if.sum()/len(df_grid_agg):.1f}%)")
print(f"Mean anomaly score: {anomaly_scores_if.mean():.4f}")

# ============================================================================
# 1.3 LOCAL OUTLIER FACTOR (LOF)
# ============================================================================
# MANTIK: Bir grid komşularına göre ne kadar "dışarıda"?
#         Global outlier değil, LOCAL outlier arar!
#
# NASIL ÇALIŞIR?
#   1. Her grid için k=20 en yakın komşuyu bul
#   2. Local Reachability Density (LRD) hesapla:
#      LRD = 1 / (ortalama mesafe komşulara)
#      - Yüksek LRD → yoğun bölge
#      - Düşük LRD → seyrek bölge
#   3. LOF skoru = (komşuların ortalama LRD'si) / (kendi LRD'si)
#      - LOF ≈ 1 → normal (komşularla aynı yoğunluk)
#      - LOF > 1 → outlier (komşulardan daha seyrek)
#
# NEDEN LOCAL?
#   - Bir grid düşük yolcu taşıyabilir ama komşuları da düşükse → normal
#   - Bir grid düşük yolcu taşıyabilir ama komşuları yüksekse → ANOMALI
#   - Local context önemli!
#
# ISOLATION FOREST ile FARK:
#   - IF global bakar, LOF local context kullanır
#   - LOF yoğunluk farkına duyarlı
# ============================================================================

print("\n" + "-" * 80)
print("1.3 LOCAL OUTLIER FACTOR (LOF)")
print("-" * 80)

lof = LocalOutlierFactor(
    n_neighbors=20,      # Her grid için 20 en yakın komşu
    contamination=TARGET_ANOMALY_RATE,   # ADIL KIYASLAMA: Aynı oran
    novelty=False        # Training data'yı test et (novelty detection değil)
)
print(f"Contamination: {TARGET_ANOMALY_RATE:.1%} (adil kıyas)")

# Model fit + predict
predictions_lof = lof.fit_predict(X_scaled)

# LOF scores (negatif değerler, daha negatif = daha outlier)
lof_scores = lof.negative_outlier_factor_

# -1 = anomaly, 1 = normal
anomalies_lof = predictions_lof == -1

df_grid_agg['lof_score'] = lof_scores
df_grid_agg['anomaly_lof'] = anomalies_lof

print(f"Neighbors: 20")
print(f"Contamination: {TARGET_ANOMALY_RATE:.1%}")
print(f"Anomalies detected: {anomalies_lof.sum()} ({100*anomalies_lof.sum()/len(df_grid_agg):.1f}%)")
print(f"Mean LOF score: {lof_scores.mean():.4f}")

# ============================================================================
# 1.4 GRAPH-BASED SPATIAL ANOMALY DETECTION
# ============================================================================
# MANTIK: Yan yana grid'ler benzer hizmet-talep dengesi göstermelidir
#         (Mekânsal süreklilik varsayımı)
#
# NASIL ÇALIŞIR?
#   1. Spatial graph oluştur:
#      - Node = her grid
#      - Edge = 1.5km içindeki komşu gridler arası bağlantı
#   2. Her grid için mekânsal komşularının ortalama özelliklerini hesapla
#   3. Grid'in kendi özellikleri ile komşu ortalamayı karşılaştır
#   4. Büyük fark = mekânsal anomali
#
# NEDEN ÖNEMLI?
#   - LOF feature space'te komşu bakar (özellik benzerligi)
#   - Graph mekânsal komşu bakar (fiziksel yakınlık)
#   - Mekânda yan yana ama feature'da farklı → sistematik uyumsuzluk
#
# ÖRNEK SENARYO:
#   Grid A (merkez): 2.6M yolcu, 1,800 sefer
#   Komşu gridler: ortalama 2.5M yolcu, 2,500 sefer
#   → Grid A daha az sefer alıyor (komşularına göre) → ANOMALI
#
# THRESHOLD:
#   - 1.5km (≈1.5 grid cell): çok yakın komşuluk
#   - Daha büyük mesafe → fazla grid bağlanır, detay kaybolur
#   - Daha küçük mesafe → çok az bağlantı, graph parçalanır
# ============================================================================

print("\n" + "-" * 80)
print("1.4 GRAPH-BASED SPATIAL ANOMALY DETECTION")
print("-" * 80)

# Spatial graph oluştur
print("Building spatial graph...")
coords = df_grid_agg[['grid_lat', 'grid_lon']].values
G = nx.Graph()

# Node'ları ekle (her grid bir node)
for i, grid_id in enumerate(df_grid_agg['grid_id']):
    G.add_node(i, grid_id=grid_id)

# Edge'leri ekle (komşu gridleri bağla)
DISTANCE_THRESHOLD = 0.015  # ≈1.5km (1.5 grid cell)
for i in range(len(coords)):
    for j in range(i+1, len(coords)):
        # Euclidean distance (lat-lon space'te)
        dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)

        # Threshold içindeyse edge ekle
        if dist < DISTANCE_THRESHOLD:
            # Weight = mesafe ile ters orantılı (yakın komşu → yüksek weight)
            G.add_edge(i, j, weight=1.0/dist)

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")

# Graph-based anomaly: grid that differs significantly from its neighbors
print("\nComputing graph-based anomalies...")
graph_anomaly_scores = []
isolated_grids = []

for i in range(len(df_grid_agg)):
    neighbors = list(G.neighbors(i))

    if len(neighbors) == 0:
        # İzole grid - geçici olarak -1 ile işaretle
        isolated_grids.append(i)
        graph_anomaly_scores.append(-1)
        continue

    # Compare this grid's features with average of neighbors
    self_features = X_scaled[i]
    neighbor_features = X_scaled[neighbors].mean(axis=0)

    # Euclidean distance
    distance = np.sqrt(np.sum((self_features - neighbor_features)**2))
    graph_anomaly_scores.append(distance)

graph_anomaly_scores = np.array(graph_anomaly_scores)

# İzole gridlere en yüksek score'dan daha yüksek değer ver
if len(isolated_grids) > 0:
    max_connected_score = graph_anomaly_scores[graph_anomaly_scores > -1].max()
    graph_anomaly_scores[graph_anomaly_scores == -1] = max_connected_score * 1.5
    print(f"Isolated grids: {len(isolated_grids)} (assigned score = {max_connected_score*1.5:.4f})")

# ADIL KIYASLAMA: Percentile-based threshold (mean+2std yerine)
threshold_graph = np.percentile(graph_anomaly_scores, PERCENTILE_THRESHOLD)
anomalies_graph = graph_anomaly_scores > threshold_graph

df_grid_agg['graph_anomaly_score'] = graph_anomaly_scores
df_grid_agg['anomaly_graph'] = anomalies_graph

print(f"Mean graph anomaly score: {graph_anomaly_scores.mean():.4f}")
print(f"Anomaly threshold (percentile): {threshold_graph:.4f}")
print(f"Anomalies detected: {anomalies_graph.sum()} ({100*anomalies_graph.sum()/len(df_grid_agg):.1f}%)")

# ----------------------------------------------------------------------------
# 1.5 CONSENSUS ANOMALY (Ensemble)
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("1.5 CONSENSUS ANOMALY DETECTION")
print("-" * 80)

# Count how many methods flagged each grid as anomaly
df_grid_agg['anomaly_count'] = (
    df_grid_agg['anomaly_autoencoder'].astype(int) +
    df_grid_agg['anomaly_isolation_forest'].astype(int) +
    df_grid_agg['anomaly_lof'].astype(int) +
    df_grid_agg['anomaly_graph'].astype(int)
)

# Consensus: flagged by at least 2 methods
df_grid_agg['anomaly_consensus'] = df_grid_agg['anomaly_count'] >= 2

print(f"Grids flagged by 0 methods: {(df_grid_agg['anomaly_count']==0).sum()}")
print(f"Grids flagged by 1 method: {(df_grid_agg['anomaly_count']==1).sum()}")
print(f"Grids flagged by 2 methods: {(df_grid_agg['anomaly_count']==2).sum()}")
print(f"Grids flagged by 3 methods: {(df_grid_agg['anomaly_count']==3).sum()}")
print(f"Grids flagged by 4 methods: {(df_grid_agg['anomaly_count']==4).sum()}")
print(f"\nConsensus anomalies (>=2 methods): {df_grid_agg['anomaly_consensus'].sum()}")

# ----------------------------------------------------------------------------
# JACCARD SIMILARITY & OVERLAP ANALYSIS (Adil Kıyaslama Kontrolü)
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ALGORITHM OVERLAP ANALYSIS (Jaccard Similarity)")
print("-" * 80)

# Jaccard similarity: |A ∩ B| / |A ∪ B|
def jaccard_similarity(set1, set2):
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union > 0 else 0

# Her algoritmanın tespit ettiği grid'leri set olarak al
anom_sets = {
    'Autoencoder': set(df_grid_agg[df_grid_agg['anomaly_autoencoder']].index),
    'IF': set(df_grid_agg[df_grid_agg['anomaly_isolation_forest']].index),
    'LOF': set(df_grid_agg[df_grid_agg['anomaly_lof']].index),
    'Graph': set(df_grid_agg[df_grid_agg['anomaly_graph']].index)
}

# Pairwise Jaccard similarity
algorithms = list(anom_sets.keys())
print("\nJaccard Similarity Matrix:")
print(f"{'':>12s}", end='')
for alg in algorithms:
    print(f"{alg:>12s}", end='')
print()

for alg1 in algorithms:
    print(f"{alg1:>12s}", end='')
    for alg2 in algorithms:
        if alg1 == alg2:
            print(f"{'1.000':>12s}", end='')
        else:
            jaccard = jaccard_similarity(anom_sets[alg1], anom_sets[alg2])
            print(f"{jaccard:>12.3f}", end='')
    print()

# Her algoritmanın unique tespit ettiği grid'ler
print("\nUnique Detections (sadece bir algoritma tespit etti):")
for alg in algorithms:
    unique = anom_sets[alg]
    for other_alg in algorithms:
        if other_alg != alg:
            unique = unique - anom_sets[other_alg]
    print(f"  {alg:>12s}: {len(unique)} grid (sadece {alg} tespit etti)")

# Tüm algoritmaların ortak tespit ettiği grid'ler
intersection_all = anom_sets['Autoencoder'] & anom_sets['IF'] & anom_sets['LOF'] & anom_sets['Graph']
print(f"\nAll 4 algorithms agree: {len(intersection_all)} grid (en güvenilir anomaliler)")

# Save anomaly results
df_grid_agg.to_csv('data/grid_anomaly_results.csv', index=False, encoding='utf-8-sig')
print("\n[SAVED] data/grid_anomaly_results.csv")

# ============================================================================
# PART 2: REGIME SHIFT ANALYSIS (TEMPORAL STABILITY)
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: REGIME SHIFT ANALYSIS")
print("=" * 80)

# ============================================================================
# 2.1 EMBEDDING STABILITY ANALYSIS
# ============================================================================
# MANTIK: Aynı grid'in günlük temsili stabil mi yoksa sürekli değişiyor mu?
#
# NASIL ÇALIŞIR?
#   1. Her gün için tüm grid'leri PCA ile 2D embedding'e dönüştür
#      → Her grid her gün bir (x,y) noktası haline gelir
#   2. Her grid için ardışık günler arası Euclidean mesafe hesapla
#      → Gün t ile gün t+1 arasındaki embedding farkı
#   3. Ortalama mesafe = instability score
#      → Yüksek mesafe = grid embedding uzayında sıçrıyor
#
# EMBEDDING NEDİR?
#   - PCA ile 7 boyutlu grid'i 2D'ye indirge
#   - Bu 2D uzay, sistemin grid'i nasıl "gördüğü"nün geometrik temsili
#   - Stabil grid → her gün aynı bölgede
#   - Unstable grid → her gün farklı yerde (kaotik trajektori)
#
# NEDEN 2D?
#   - Görselleştirme için
#   - İlk 2 PC varyansın çoğunu yakalıyor
#   - Distance hesaplamak kolay
#
# YORUMLAMA:
#   Düşük instability → Grid sistemin temsil uzayında stabil
#   Yüksek instability → Sistem grid'i tutarlı okuyamıyor → REJİM KAYMASI
#
# MAKİNE SEMİYOTİĞİ:
#   Embedding uzayı = sistemin grid'leri nasıl kategorize ettiğinin geometrisi
#   Instability = sistemin epistemolojik tutarsızlığı
#   "Aynı mekân, farklı günlerde farklı şekilde okunuyor"
# ============================================================================

print("\n" + "-" * 80)
print("2.1 EMBEDDING STABILITY ANALYSIS")
print("-" * 80)

print("Computing daily embeddings for each grid...")

# Tüm tarihleri ve grid ID'leri al
dates = sorted(df['tarih'].unique())
grid_ids = sorted(df['grid_id'].unique())

# Her tarih için embedding hesapla
# embeddings_by_date = {date: numpy array (1184 × 2)}
embeddings_by_date = {}

for date in dates:
    # O güne ait veriyi al
    df_date = df[df['tarih'] == date]

    # Tüm grid'lere karşılık gelen satırları oluştur
    # Bazı gridler o gün veri olmayabilir → fillna(0) ile doldur
    df_date_merged = df_date.set_index('grid_id').reindex(grid_ids)

    # Feature matrix (eksik gridler için 0)
    X_date = df_date_merged[feature_cols].fillna(0).values

    # Standardize (aynı scaler ile, fit etme!)
    X_date_scaled = scaler.transform(X_date)

    # PCA embedding: ilk 2 principal component
    # (1184 × 7) → (1184 × 2)
    embedding = pca.transform(X_date_scaled)[:, :2]

    embeddings_by_date[date] = embedding

print(f"Embeddings computed for {len(dates)} dates")

# Compute embedding stability for each grid
# Stability = average cosine similarity between consecutive days
print("\nComputing embedding stability...")

stability_scores = []

for i, grid_id in enumerate(grid_ids):
    embeddings_this_grid = [embeddings_by_date[date][i] for date in dates]

    # Compute pairwise distances between consecutive days
    distances = []
    for j in range(len(embeddings_this_grid) - 1):
        emb1 = embeddings_this_grid[j]
        emb2 = embeddings_this_grid[j+1]

        # Use Euclidean distance
        dist = euclidean(emb1, emb2)
        distances.append(dist)

    # Average distance = instability
    avg_distance = np.mean(distances) if distances else 0
    stability_scores.append(avg_distance)

stability_scores = np.array(stability_scores)

# Create grid stability dataframe
df_stability = pd.DataFrame({
    'grid_id': grid_ids,
    'embedding_instability': stability_scores
})

# Merge with grid info
df_stability = df_stability.merge(
    df_grid_agg[['grid_id', 'grid_lat', 'grid_lon']],
    on='grid_id',
    how='left'
)

# High instability = regime shift
threshold_instability = stability_scores.mean() + 1.5 * stability_scores.std()
df_stability['regime_shift_embedding'] = stability_scores > threshold_instability

print(f"Mean embedding instability: {stability_scores.mean():.4f}")
print(f"Std embedding instability: {stability_scores.std():.4f}")
print(f"Instability threshold: {threshold_instability:.4f}")
print(f"Grids with regime shift: {df_stability['regime_shift_embedding'].sum()} ({100*df_stability['regime_shift_embedding'].sum()/len(df_stability):.1f}%)")

# ============================================================================
# 2.2 DAILY CLUSTERING REGIME SHIFT
# ============================================================================
# MANTIK: Sistem her gün grid'leri kategorilere (cluster) ayırıyor.
#         Aynı grid sürekli kategori değiştiriyorsa → kategorik istikrarsızlık
#
# NASIL ÇALIŞIR?
#   1. TÜM veriye GLOBAL K-Means clustering fit et (k=5)
#      → Sistem genelinde 5 sabit "tip" (cluster) tanımla
#   2. Her gün için bu SABIT model ile cluster ata
#      → Aynı model, her gün sadece predict
#      → Cluster 0 her gün AYNI anlama gelir (centroid sabit)
#   3. Her grid için cluster switching rate hesapla
#      → Kaç gün cluster değiştirdi / toplam gün sayısı
#   4. Yüksek switch rate = grid GERÇEKTEN kategori değiştiriyor
#      → Model artifact değil, asıl regime shift
#
# CLUSTER'LAR NE ANLAMA GELİR?
#   Sistem grid'leri otomatik olarak kategorize ediyor:
#     Cluster 0: "Düşük talep + Düşük hizmet"
#     Cluster 1: "Yüksek talep + Yüksek hizmet"
#     Cluster 2: "Yüksek talep + Düşük hizmet" (problem!)
#     Cluster 3: "Düşük talep + Yüksek hizmet" (over-service)
#     Cluster 4: "Orta düzey"
#
# NEDEN 5 CLUSTER?
#   - 3 → çok az, detay kaybı
#   - 10 → çok fazla, over-fitting
#   - 5 → dengeli (tipik sistem durumları)
#
# SWITCHING RATE YORUMU:
#   Düşük rate (örn: 30%) → Grid hep benzer cluster'da (stabil tip)
#   Yüksek rate (örn: 80%) → Grid sürekli cluster değiştiriyor (istikrarsız)
#
# EMBEDDING STABILITY ile FARK:
#   - Embedding: Geometrik pozisyon değişimi (sürekli)
#   - Clustering: Kategorik tip değişimi (kesikli)
#   - Embedding stabil ama clustering unstable olabilir!
#     → Grid embedding uzayında aynı yerde ama cluster sınırında
#
# MAKİNE SEMİYOTİĞİ:
#   Sistem grid'i her gün farklı bir "tipe" atıyor
#   → Grid sistemin kategorik çerçevesine sığmıyor
#   → Eşik problemi: Grid cluster sınırlarında dalgalanıyor
# ============================================================================

print("\n" + "-" * 80)
print("2.2 DAILY CLUSTERING REGIME SHIFT")
print("-" * 80)

# K-Means cluster sayısı
n_clusters = 5

# ============================================================================
# IMPROVED APPROACH: Global Clustering (FIT ONCE, PREDICT DAILY)
# ============================================================================
# OLD APPROACH (Hungarian alignment):
#   - Fit K-Means every day → different centroids each day
#   - Align labels with Hungarian → helps but still has drift
#   - Result: 41% switch rate (still high!)
#
# NEW APPROACH (Global clustering):
#   - Fit K-Means ONCE on all data → fixed centroids
#   - Predict cluster for each day using same model
#   - Result: True regime shifts, no model artifacts
# ============================================================================

print("Fitting global K-Means model on all data...")

# Prepare all data (all grids × all dates)
X_all_dates = []
for date in dates:
    df_date = df[df['tarih'] == date]
    df_date_merged = df_date.set_index('grid_id').reindex(grid_ids)
    X_date = df_date_merged[feature_cols].fillna(0).values
    X_date_scaled = scaler.transform(X_date)
    X_all_dates.append(X_date_scaled)

X_all_dates = np.array(X_all_dates)  # Shape: (173 dates, 1184 grids, 6 features)

# Reshape to (173 * 1184, 6) for clustering
X_all_flat = X_all_dates.reshape(-1, X_all_dates.shape[2])

print(f"  Total samples: {len(X_all_flat):,} (dates × grids)")
print(f"  Fitting K-Means with {n_clusters} clusters...")

# Fit global K-Means model
kmeans_global = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=20,  # More initializations for better global optimum
    max_iter=500
)
kmeans_global.fit(X_all_flat)

print(f"  Global model fitted!")
print(f"  Cluster sizes: {np.bincount(kmeans_global.labels_)}")

# Now predict cluster for each date using the SAME model
print("\nPredicting clusters for each day using global model...")

cluster_assignments = {}

for date_idx, date in enumerate(dates):
    X_date_scaled = X_all_dates[date_idx]

    # Predict using global model (NO REFIT!)
    labels = kmeans_global.predict(X_date_scaled)

    cluster_assignments[date] = labels

print(f"Clustering completed for {len(dates)} dates")

# Compute cluster switching frequency for each grid
print("\nComputing cluster switching frequency...")

cluster_switches = []

for i, grid_id in enumerate(grid_ids):
    labels_this_grid = [cluster_assignments[date][i] for date in dates]

    # Count how many times cluster changed
    switches = 0
    for j in range(len(labels_this_grid) - 1):
        if labels_this_grid[j] != labels_this_grid[j+1]:
            switches += 1

    # Switching rate = switches / (days - 1)
    switch_rate = switches / (len(dates) - 1) if len(dates) > 1 else 0
    cluster_switches.append(switch_rate)

cluster_switches = np.array(cluster_switches)

df_stability['cluster_switch_rate'] = cluster_switches

# High switch rate = regime instability
threshold_switch = cluster_switches.mean() + 1.5 * cluster_switches.std()
df_stability['regime_shift_clustering'] = cluster_switches > threshold_switch

print(f"Mean cluster switch rate: {cluster_switches.mean():.2%}")
print(f"Std cluster switch rate: {cluster_switches.std():.2%}")
print(f"Switch rate threshold: {threshold_switch:.2%}")
print(f"Grids with regime shift: {df_stability['regime_shift_clustering'].sum()} ({100*df_stability['regime_shift_clustering'].sum()/len(df_stability):.1f}%)")

# ----------------------------------------------------------------------------
# 2.3 CONSENSUS REGIME SHIFT
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("2.3 CONSENSUS REGIME SHIFT")
print("-" * 80)

df_stability['regime_shift_consensus'] = (
    df_stability['regime_shift_embedding'] |
    df_stability['regime_shift_clustering']
)

print(f"Grids with regime shift (either method): {df_stability['regime_shift_consensus'].sum()}")
print(f"Grids with regime shift (both methods): {(df_stability['regime_shift_embedding'] & df_stability['regime_shift_clustering']).sum()}")

# Save stability results
df_stability.to_csv('data/grid_regime_shift_results.csv', index=False, encoding='utf-8-sig')
print("\n[SAVED] data/grid_regime_shift_results.csv")

# ============================================================================
# PART 3: COMBINED ANALYSIS & INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: COMBINED ANALYSIS")
print("=" * 80)

# Merge anomaly and stability results
df_combined = df_grid_agg.merge(df_stability, on='grid_id', how='left')

# Identify critical grids: both anomalous AND unstable
df_combined['critical_grid'] = (
    df_combined['anomaly_consensus'] &
    df_combined['regime_shift_consensus']
)

print(f"\nCritical grids (anomalous + unstable): {df_combined['critical_grid'].sum()}")

# Save combined results
df_combined.to_csv('data/grid_combined_analysis.csv', index=False, encoding='utf-8-sig')
print("[SAVED] data/grid_combined_analysis.csv")

# Top 20 critical grids
print("\nTop 20 Most Critical Grids:")
df_critical = df_combined[df_combined['critical_grid']].copy()
df_critical['criticality_score'] = (
    df_critical['anomaly_count'] +
    2 * df_critical['embedding_instability'] +
    2 * df_critical['cluster_switch_rate']
)
df_critical_top20 = df_critical.nlargest(20, 'criticality_score')

for idx, row in df_critical_top20.iterrows():
    print(f"  {row['grid_id']}: "
          f"Anomaly methods={row['anomaly_count']}, "
          f"Instability={row['embedding_instability']:.3f}, "
          f"Switch rate={row['cluster_switch_rate']:.1%}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  1. data/grid_anomaly_results.csv - Anomaly detection results")
print("  2. data/grid_regime_shift_results.csv - Regime shift results")
print("  3. data/grid_combined_analysis.csv - Combined analysis")
