"""
Normalize and compare metro + bus passenger volumes
Show relative density/usage across all transit modes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("NORMALIZING METRO + BUS PASSENGER DATA")
print("=" * 70)

# 1. Load metro data
print("\n1. Loading metro data...")
df_metro = pd.read_csv('data/ego_metro_data_with_dates.csv', encoding='utf-8-sig')
df_metro['tarih'] = pd.to_datetime(df_metro['tarih'])

# Average daily passengers per metro line
metro_avg = df_metro.groupby('hat_no').agg({
    'tasinan_yolcu': 'mean',
    'doluluk_orani': 'mean',
    'sefer_sayisi': 'mean'
}).reset_index()

metro_avg.columns = ['hat_no', 'avg_yolcu', 'avg_doluluk', 'avg_sefer']
metro_avg['mod'] = 'Metro'

print(f"   Metro lines: {len(metro_avg)}")
print(f"   Total metro avg passengers/day: {metro_avg['avg_yolcu'].sum():,.0f}")

# 2. Load bus data
print("\n2. Loading bus data...")
df_bus = pd.read_csv('data/ego_data_with_dates_CORRECTED.csv', encoding='utf-8-sig')
df_bus['TARIH'] = pd.to_datetime(df_bus['TARIH'])

# Average daily passengers per bus line
bus_avg = df_bus.groupby('HAT NO').agg({
    'TAŞINAN YOLCU SAYISI': 'mean',
    'DOLULUK ORANI': 'mean',
    'SEFER SAYISI': 'mean'
}).reset_index()

bus_avg.columns = ['hat_no', 'avg_yolcu', 'avg_doluluk', 'avg_sefer']
bus_avg['hat_no'] = bus_avg['hat_no'].astype(str)
bus_avg['mod'] = 'Otobüs'

print(f"   Bus routes: {len(bus_avg)}")
print(f"   Total bus avg passengers/day: {bus_avg['avg_yolcu'].sum():,.0f}")

# 3. Combine all transit
print("\n3. Combining all transit modes...")
df_all = pd.concat([metro_avg, bus_avg], ignore_index=True)

# Calculate overall statistics
overall_mean = df_all['avg_yolcu'].mean()
overall_std = df_all['avg_yolcu'].std()
overall_median = df_all['avg_yolcu'].median()

print(f"\nOverall statistics (all lines):")
print(f"   Mean passengers/day: {overall_mean:,.0f}")
print(f"   Median passengers/day: {overall_median:,.0f}")
print(f"   Std deviation: {overall_std:,.0f}")

# 4. Normalize using z-score
df_all['normalized_yolcu'] = (df_all['avg_yolcu'] - overall_mean) / overall_std

# Also calculate percentile rank (0-100)
df_all['percentile'] = df_all['avg_yolcu'].rank(pct=True) * 100

# Categorize lines
def categorize_usage(percentile):
    if percentile >= 95:
        return 'Çok Yoğun (>95%)'
    elif percentile >= 75:
        return 'Yoğun (75-95%)'
    elif percentile >= 50:
        return 'Orta (50-75%)'
    elif percentile >= 25:
        return 'Az (25-50%)'
    else:
        return 'Çok Az (<25%)'

df_all['category'] = df_all['percentile'].apply(categorize_usage)

# Save normalized data
df_all.to_csv('data/normalized_all_transit.csv', index=False, encoding='utf-8-sig')
print(f"\n[SAVED] data/normalized_all_transit.csv")

# 5. Create visualizations
print("\n4. Creating visualizations...")

# Plot 1: Top 30 lines (metro + bus combined)
fig, ax = plt.subplots(figsize=(14, 10))

top30 = df_all.nlargest(30, 'avg_yolcu')
colors = ['red' if m == 'Metro' else 'steelblue' for m in top30['mod']]

ax.barh(range(len(top30)), top30['avg_yolcu'], color=colors)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels([f"{row['hat_no']} ({row['mod']})" for _, row in top30.iterrows()])
ax.set_xlabel('Ortalama Günlük Yolcu Sayısı')
ax.set_title('En Yoğun 30 Hat (Metro + Otobüs)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add average line
ax.axvline(x=overall_mean, color='green', linestyle='--', linewidth=2, label=f'Genel Ortalama: {overall_mean:,.0f}')
ax.legend()

plt.tight_layout()
plt.savefig('maps/top30_all_transit.png', dpi=300, bbox_inches='tight')
print("   [SAVED] maps/top30_all_transit.png")
plt.close()

# Plot 2: Distribution comparison (Metro vs Bus)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
metro_data = df_all[df_all['mod'] == 'Metro']['avg_yolcu']
bus_data = df_all[df_all['mod'] == 'Otobüs']['avg_yolcu']

ax1.hist(bus_data, bins=50, alpha=0.7, color='steelblue', label='Otobüs', edgecolor='black')
ax1.hist(metro_data, bins=10, alpha=0.7, color='red', label='Metro', edgecolor='black')
ax1.axvline(x=overall_mean, color='green', linestyle='--', linewidth=2, label='Genel Ortalama')
ax1.set_xlabel('Ortalama Günlük Yolcu')
ax1.set_ylabel('Hat Sayısı')
ax1.set_title('Yolcu Dağılımı (Metro vs Otobüs)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
data_to_plot = [metro_data, bus_data]
ax2.boxplot(data_to_plot, labels=['Metro', 'Otobüs'], patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red', linewidth=2))
ax2.axhline(y=overall_mean, color='green', linestyle='--', linewidth=2, label='Genel Ortalama')
ax2.set_ylabel('Ortalama Günlük Yolcu')
ax2.set_title('Yolcu Dağılımı (Box Plot)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('maps/transit_distribution.png', dpi=300, bbox_inches='tight')
print("   [SAVED] maps/transit_distribution.png")
plt.close()

# Plot 3: Normalized comparison (z-scores)
fig, ax = plt.subplots(figsize=(14, 10))

# Top 30 by normalized score
top30_norm = df_all.nlargest(30, 'normalized_yolcu')
colors = ['red' if m == 'Metro' else 'steelblue' for m in top30_norm['mod']]

ax.barh(range(len(top30_norm)), top30_norm['normalized_yolcu'], color=colors)
ax.set_yticks(range(len(top30_norm)))
ax.set_yticklabels([f"{row['hat_no']} ({row['mod']})" for _, row in top30_norm.iterrows()])
ax.set_xlabel('Normalize Edilmiş Yolcu Skoru (Z-Score)')
ax.set_title('En Yoğun 30 Hat (Normalize Edilmiş)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Ortalama (0)')
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

plt.tight_layout()
plt.savefig('maps/top30_normalized.png', dpi=300, bbox_inches='tight')
print("   [SAVED] maps/top30_normalized.png")
plt.close()

# Plot 4: Category distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Category counts by mode
category_counts = df_all.groupby(['category', 'mod']).size().unstack(fill_value=0)
category_order = ['Çok Yoğun (>95%)', 'Yoğun (75-95%)', 'Orta (50-75%)', 'Az (25-50%)', 'Çok Az (<25%)']
category_counts = category_counts.reindex(category_order)

category_counts.plot(kind='bar', stacked=False, ax=ax1, color=['red', 'steelblue'])
ax1.set_xlabel('Kategori')
ax1.set_ylabel('Hat Sayısı')
ax1.set_title('Yoğunluk Kategorisi Dağılımı', fontweight='bold')
ax1.legend(title='Mod')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# Pie chart - total by mode
mode_totals = df_all.groupby('mod')['avg_yolcu'].sum()
ax2.pie(mode_totals, labels=mode_totals.index, autopct='%1.1f%%',
        colors=['red', 'steelblue'], startangle=90)
ax2.set_title('Toplam Yolcu Payı (Metro vs Otobüs)', fontweight='bold')

plt.tight_layout()
plt.savefig('maps/transit_categories.png', dpi=300, bbox_inches='tight')
print("   [SAVED] maps/transit_categories.png")
plt.close()

# 6. Statistics summary
print(f"\n{'=' * 70}")
print("NORMALIZED STATISTICS SUMMARY")
print("=" * 70)

print(f"\nMetro:")
metro_stats = df_all[df_all['mod'] == 'Metro']
print(f"   Hatlar: {len(metro_stats)}")
print(f"   Toplam yolcu/gün: {metro_stats['avg_yolcu'].sum():,.0f}")
print(f"   Ortalama yolcu/hat: {metro_stats['avg_yolcu'].mean():,.0f}")
print(f"   Genel ortalamanın üstünde: {(metro_stats['avg_yolcu'] > overall_mean).sum()} hat")

print(f"\nOtobüs:")
bus_stats = df_all[df_all['mod'] == 'Otobüs']
print(f"   Hatlar: {len(bus_stats)}")
print(f"   Toplam yolcu/gün: {bus_stats['avg_yolcu'].sum():,.0f}")
print(f"   Ortalama yolcu/hat: {bus_stats['avg_yolcu'].mean():,.0f}")
print(f"   Genel ortalamanın üstünde: {(bus_stats['avg_yolcu'] > overall_mean).sum()} hat")

print(f"\nToplam:")
print(f"   Tüm hatlar: {len(df_all)}")
print(f"   Toplam yolcu/gün: {df_all['avg_yolcu'].sum():,.0f}")
print(f"   Genel ortalama: {overall_mean:,.0f}")

print(f"\nEn yoğun 10 hat (tüm modlar):")
for idx, row in df_all.nlargest(10, 'avg_yolcu').iterrows():
    print(f"   {row['hat_no']:10s} ({row['mod']:7s}): {row['avg_yolcu']:>10,.0f} yolcu/gün (Persentil: {row['percentile']:.1f}%)")

print(f"\n{'=' * 70}")
print("NORMALIZATION COMPLETE")
print("=" * 70)
print("\nCreated visualizations:")
print("  1. maps/top30_all_transit.png - En yoğun 30 hat")
print("  2. maps/transit_distribution.png - Dağılım karşılaştırması")
print("  3. maps/top30_normalized.png - Normalize edilmiş top 30")
print("  4. maps/transit_categories.png - Kategori dağılımı")
