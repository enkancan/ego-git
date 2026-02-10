"""
DAĞITIM YÖNTEMLERİNİ KARŞILAŞTIRMA

İki farklı dağıtım yöntemini karşılaştırır:
1. Connectivity-Based (Transfer Gücü)
2. Position-Based (Terminal-Biased)

Çıktılar:
- İstatistiksel karşılaştırma
- Scatter plots
- Histogram (fark dağılımı)
- Korelasyon analizi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Stil ayarları
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Klasör oluştur
OUTPUT_DIR = Path('data/comparison')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("DISTRIBUTION METHOD COMPARISON")
print("=" * 70)

# ============================================================================
# 1. VERİLERİ YÜKLE
# ============================================================================
print("\n1. Loading data...")

df_conn = pd.read_csv('data/daily_grid_data.csv')
df_pos = pd.read_csv('data/daily_grid_data_position.csv')

print(f"   Connectivity-based: {len(df_conn):,} records")
print(f"   Position-based: {len(df_pos):,} records")

# ============================================================================
# 2. VERİLERİ BİRLEŞTİR
# ============================================================================
print("\n2. Merging datasets...")

# Her iki veriyi tarih ve grid_id'ye göre birleştir
df_merged = df_conn.merge(
    df_pos,
    on=['tarih', 'grid_id', 'grid_lat', 'grid_lon'],
    suffixes=('_conn', '_pos')
)

print(f"   Merged records: {len(df_merged):,}")
print(f"   Matching grids: {df_merged['grid_id'].nunique()}")
print(f"   Matching dates: {df_merged['tarih'].nunique()}")

# ============================================================================
# 3. FARKLAR HESAPLA
# ============================================================================
print("\n3. Calculating differences...")

# Mutlak farklar
df_merged['diff_yolcu'] = df_merged['yolcu_per_stop_pos'] - df_merged['yolcu_per_stop_conn']
df_merged['diff_sefer'] = df_merged['sefer_per_stop_pos'] - df_merged['sefer_per_stop_conn']
df_merged['diff_kapasite'] = df_merged['kapasite_per_stop_pos'] - df_merged['kapasite_per_stop_conn']
df_merged['diff_doluluk'] = df_merged['doluluk_orani_pos'] - df_merged['doluluk_orani_conn']

# Yüzdesel farklar
df_merged['pct_diff_yolcu'] = (
    df_merged['diff_yolcu'] / df_merged['yolcu_per_stop_conn'].replace(0, np.nan)
) * 100

df_merged['pct_diff_sefer'] = (
    df_merged['diff_sefer'] / df_merged['sefer_per_stop_conn'].replace(0, np.nan)
) * 100

df_merged['pct_diff_kapasite'] = (
    df_merged['diff_kapasite'] / df_merged['kapasite_per_stop_conn'].replace(0, np.nan)
) * 100

# ============================================================================
# 4. İSTATİSTİKSEL ANALİZ
# ============================================================================
print("\n4. Statistical analysis...")

stats = {}

for metric in ['yolcu', 'sefer', 'kapasite', 'doluluk']:
    diff_col = f'diff_{metric}'
    conn_col = f'{metric}_per_stop_conn' if metric != 'doluluk' else 'doluluk_orani_conn'
    pos_col = f'{metric}_per_stop_pos' if metric != 'doluluk' else 'doluluk_orani_pos'

    # Temel istatistikler
    mean_diff = df_merged[diff_col].mean()
    std_diff = df_merged[diff_col].std()
    min_diff = df_merged[diff_col].min()
    max_diff = df_merged[diff_col].max()
    median_diff = df_merged[diff_col].median()

    # Korelasyon
    pearson_r, pearson_p = pearsonr(df_merged[conn_col], df_merged[pos_col])
    spearman_r, spearman_p = spearmanr(df_merged[conn_col], df_merged[pos_col])

    # Ortalama değerler
    mean_conn = df_merged[conn_col].mean()
    mean_pos = df_merged[pos_col].mean()

    stats[metric] = {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'min_diff': min_diff,
        'max_diff': max_diff,
        'median_diff': median_diff,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mean_conn': mean_conn,
        'mean_pos': mean_pos,
        'pct_change': ((mean_pos - mean_conn) / mean_conn) * 100 if mean_conn != 0 else 0
    }

    print(f"\n   {metric.upper()}:")
    print(f"     Connectivity mean: {mean_conn:.2f}")
    print(f"     Position mean: {mean_pos:.2f}")
    print(f"     Mean difference: {mean_diff:.2f} ({stats[metric]['pct_change']:.1f}%)")
    print(f"     Std difference: {std_diff:.2f}")
    print(f"     Range: [{min_diff:.2f}, {max_diff:.2f}]")
    print(f"     Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"     Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")

# ============================================================================
# 5. VİZUALİZASYONLAR
# ============================================================================
print("\n5. Creating visualizations...")

# 5.1 Scatter Plots
print("   Creating scatter plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Connectivity-Based vs Position-Based: Scatter Comparison', fontsize=16, fontweight='bold')

metrics_plot = [
    ('yolcu_per_stop', 'Passengers per Stop'),
    ('sefer_per_stop', 'Trips per Stop'),
    ('kapasite_per_stop', 'Capacity per Stop'),
    ('doluluk_orani', 'Occupancy Rate (%)')
]

for idx, (metric, label) in enumerate(metrics_plot):
    ax = axes[idx // 2, idx % 2]

    conn_col = f'{metric}_conn'
    pos_col = f'{metric}_pos'

    # Scatter plot
    ax.scatter(df_merged[conn_col], df_merged[pos_col], alpha=0.3, s=10)

    # Perfect correlation line (y=x)
    min_val = min(df_merged[conn_col].min(), df_merged[pos_col].min())
    max_val = max(df_merged[conn_col].max(), df_merged[pos_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match (y=x)')

    # Korelasyon bilgisi
    metric_key = metric.replace('_per_stop', '').replace('_orani', '')
    r = stats[metric_key]['pearson_r']

    ax.set_xlabel(f'Connectivity-Based {label}', fontsize=11)
    ax.set_ylabel(f'Position-Based {label}', fontsize=11)
    ax.set_title(f'{label}\nPearson r = {r:.4f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'scatter_comparison.png', dpi=300, bbox_inches='tight')
print(f"     [SAVED] {OUTPUT_DIR / 'scatter_comparison.png'}")
plt.close()

# 5.2 Difference Histograms
print("   Creating difference histograms...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Distribution of Differences (Position - Connectivity)', fontsize=16, fontweight='bold')

diff_metrics = [
    ('diff_yolcu', 'Passengers per Stop'),
    ('diff_sefer', 'Trips per Stop'),
    ('diff_kapasite', 'Capacity per Stop'),
    ('diff_doluluk', 'Occupancy Rate (%)')
]

for idx, (diff_col, label) in enumerate(diff_metrics):
    ax = axes[idx // 2, idx % 2]

    # Histogram
    ax.hist(df_merged[diff_col], bins=50, alpha=0.7, edgecolor='black')

    # Ortalama çizgisi
    mean_val = df_merged[diff_col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')

    # Sıfır çizgisi
    ax.axvline(0, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Zero Difference')

    ax.set_xlabel(f'Difference in {label}', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label} Difference Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'difference_histograms.png', dpi=300, bbox_inches='tight')
print(f"     [SAVED] {OUTPUT_DIR / 'difference_histograms.png'}")
plt.close()

# 5.3 Percentage Difference Distribution
print("   Creating percentage difference plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Percentage Differences (Position vs Connectivity)', fontsize=16, fontweight='bold')

pct_metrics = [
    ('pct_diff_yolcu', 'Passengers', 'yolcu'),
    ('pct_diff_sefer', 'Trips', 'sefer'),
    ('pct_diff_kapasite', 'Capacity', 'kapasite')
]

for idx, (pct_col, label, metric_key) in enumerate(pct_metrics):
    ax = axes[idx]

    # Outlier'ları filtrele (-200%, +200% dışındakileri çıkar)
    data = df_merged[pct_col].dropna()
    data_filtered = data[(data >= -200) & (data <= 200)]

    ax.hist(data_filtered, bins=50, alpha=0.7, edgecolor='black')

    mean_val = data.mean()
    median_val = data.median()

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.1f}%')
    ax.axvline(median_val, color='blue', linestyle=':', linewidth=2, label=f'Median = {median_val:.1f}%')
    ax.axvline(0, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Zero')

    ax.set_xlabel(f'% Difference in {label}', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label} % Difference\n(filtered: -200% to +200%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'percentage_differences.png', dpi=300, bbox_inches='tight')
print(f"     [SAVED] {OUTPUT_DIR / 'percentage_differences.png'}")
plt.close()

# ============================================================================
# 6. EN BÜYÜK FARKLAR (TOP 20)
# ============================================================================
print("\n6. Finding largest differences...")

# Mutlak fark bazında en büyük 20
df_merged['abs_diff_yolcu'] = df_merged['diff_yolcu'].abs()
top20_diff = df_merged.nlargest(20, 'abs_diff_yolcu')[
    ['tarih', 'grid_id', 'grid_lat', 'grid_lon',
     'yolcu_per_stop_conn', 'yolcu_per_stop_pos', 'diff_yolcu', 'pct_diff_yolcu']
]

print("\n   Top 20 Largest Absolute Differences (Passengers):")
print(top20_diff.to_string(index=False))

# CSV'ye kaydet
top20_diff.to_csv(OUTPUT_DIR / 'top20_largest_differences.csv', index=False)
print(f"\n     [SAVED] {OUTPUT_DIR / 'top20_largest_differences.csv'}")

# ============================================================================
# 7. ÖZET RAPOR KAYDET
# ============================================================================
print("\n7. Saving summary statistics...")

summary_df = pd.DataFrame(stats).T
summary_df.to_csv(OUTPUT_DIR / 'statistical_summary.csv')
print(f"     [SAVED] {OUTPUT_DIR / 'statistical_summary.csv'}")

# ============================================================================
# 8. SONUÇ
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print("\n1. KORELASYON:")
for metric in ['yolcu', 'sefer', 'kapasite']:
    r = stats[metric]['pearson_r']
    quality = "Çok Yüksek" if r > 0.9 else "Yüksek" if r > 0.8 else "Orta" if r > 0.7 else "Düşük"
    print(f"   {metric.capitalize()}: r = {r:.4f} ({quality})")

print("\n2. ORTALAMA DEĞİŞİM:")
for metric in ['yolcu', 'sefer', 'kapasite']:
    pct = stats[metric]['pct_change']
    direction = "artış" if pct > 0 else "azalış"
    print(f"   {metric.capitalize()}: %{abs(pct):.1f} {direction}")

print("\n3. SONUC:")
avg_correlation = np.mean([stats[m]['pearson_r'] for m in ['yolcu', 'sefer', 'kapasite']])
avg_spearman = np.mean([stats[m]['spearman_r'] for m in ['yolcu', 'sefer', 'kapasite']])
if avg_correlation > 0.85:
    print("   [OK] Yuksek korelasyon: Iki yontem benzer sonuclar veriyor")
    print("   [OK] Her iki yontem de kullanilabilir")
else:
    print(f"   [!] Pearson korelasyon: {avg_correlation:.3f} (Dusuk)")
    print(f"   [!] Spearman korelasyon: {avg_spearman:.3f} (Yuksek)")
    print("   [>>] Ranking benzer ama degerler farkli")
    print("   [>>] Position-based daha yuksek degerler veriyor")

print(f"\nTüm çıktılar: {OUTPUT_DIR}/")
print("=" * 70)
