"""
DAĞITIM YÖNTEMLERİ FARK HARİTASI

İki dağıtım yöntemi arasındaki farkları görselleştirir:
- Connectivity-based vs Position-based
- Kırmızı = Position daha yüksek
- Mavi = Connectivity daha yüksek
- Yeşil = Benzer
"""

import pandas as pd
import folium
from folium import plugins
import numpy as np
from pathlib import Path

# Klasör oluştur
OUTPUT_DIR = Path(__file__).parent.parent / "maps"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("DISTRIBUTION DIFFERENCE MAP GENERATOR")
print("=" * 70)

# ============================================================================
# 1. VERİLERİ YÜKLE
# ============================================================================
print("\n1. Loading data...")

df_conn = pd.read_csv('data/daily_grid_data.csv')
df_pos = pd.read_csv('data/daily_grid_data_position.csv')

print(f"   Connectivity: {len(df_conn):,} records")
print(f"   Position: {len(df_pos):,} records")

# ============================================================================
# 2. GRID ORTALAMALARINI HESAPLA
# ============================================================================
print("\n2. Calculating grid averages...")

conn_avg = df_conn.groupby(['grid_lat', 'grid_lon']).agg({
    'yolcu_per_stop': 'mean',
    'sefer_per_stop': 'mean',
    'kapasite_per_stop': 'mean',
    'stop_count': 'mean'
}).reset_index()

pos_avg = df_pos.groupby(['grid_lat', 'grid_lon']).agg({
    'yolcu_per_stop': 'mean',
    'sefer_per_stop': 'mean',
    'kapasite_per_stop': 'mean',
    'stop_count': 'mean'
}).reset_index()

print(f"   Connectivity grids: {len(conn_avg)}")
print(f"   Position grids: {len(pos_avg)}")

# ============================================================================
# 3. BİRLEŞTİR VE FARKLAR HESAPLA
# ============================================================================
print("\n3. Merging and calculating differences...")

df_diff = conn_avg.merge(
    pos_avg,
    on=['grid_lat', 'grid_lon'],
    suffixes=('_conn', '_pos')
)

# Mutlak ve yüzdesel farklar
df_diff['diff_yolcu'] = df_diff['yolcu_per_stop_pos'] - df_diff['yolcu_per_stop_conn']
df_diff['pct_diff_yolcu'] = (
    df_diff['diff_yolcu'] / df_diff['yolcu_per_stop_conn'].replace(0, np.nan)
) * 100

df_diff['abs_diff_yolcu'] = df_diff['diff_yolcu'].abs()

print(f"   Merged grids: {len(df_diff)}")
print(f"   Mean difference: {df_diff['diff_yolcu'].mean():.2f}")
print(f"   Median difference: {df_diff['diff_yolcu'].median():.2f}")
print(f"   Max positive diff: {df_diff['diff_yolcu'].max():.2f}")
print(f"   Max negative diff: {df_diff['diff_yolcu'].min():.2f}")

# ============================================================================
# 4. HARİTA OLUŞTUR
# ============================================================================
print("\n4. Creating difference map...")

ANKARA_CENTER = [39.9334, 32.8597]
GRID_SIZE_LAT = 1000 / 111000
GRID_SIZE_LON = 1000 / 85000

m = folium.Map(
    location=ANKARA_CENTER,
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Renk fonksiyonu: yüzde farka göre
def get_difference_color(pct_diff):
    """
    Yüzde farka göre renk döndür
    Mavi: Connectivity daha yüksek (negatif fark)
    Yeşil: Benzer (±10%)
    Kırmızı: Position daha yüksek (pozitif fark)
    """
    if pd.isna(pct_diff):
        return '#CCCCCC'  # Gri (veri yok)

    if pct_diff < -10:
        # Connectivity daha yüksek - Mavi tonları
        intensity = min(abs(pct_diff) / 100, 1.0)
        r = int(50 * (1 - intensity))
        g = int(100 * (1 - intensity))
        b = int(150 + 105 * intensity)
        return f'#%02x%02x%02x' % (r, g, b)
    elif pct_diff > 10:
        # Position daha yüksek - Kırmızı tonları
        intensity = min(pct_diff / 100, 1.0)
        r = int(150 + 105 * intensity)
        g = int(100 * (1 - intensity))
        b = int(50 * (1 - intensity))
        return f'#%02x%02x%02x' % (r, g, b)
    else:
        # Benzer - Yeşil tonları
        return '#00CC00'

# Her grid için rectangle çiz
for idx, row in df_diff.iterrows():
    grid_lat = row['grid_lat']
    grid_lon = row['grid_lon']

    conn_val = row['yolcu_per_stop_conn']
    pos_val = row['yolcu_per_stop_pos']
    diff = row['diff_yolcu']
    pct_diff = row['pct_diff_yolcu']

    # Grid sınırları
    bounds = [
        [grid_lat, grid_lon],
        [grid_lat + GRID_SIZE_LAT, grid_lon + GRID_SIZE_LON]
    ]

    # Renk
    color = get_difference_color(pct_diff)

    # Popup bilgisi
    popup_html = f"""
    <div style="font-family: Arial; font-size: 12px; min-width: 250px;">
        <h4 style="margin:0 0 10px 0; color: #333;">Dağıtım Farkı Analizi</h4>

        <b>Connectivity-Based:</b> {conn_val:.2f} yolcu/durak<br>
        <b>Position-Based:</b> {pos_val:.2f} yolcu/durak<br>
        <hr style="margin:5px 0;">
        <b>Fark:</b> {diff:.2f} ({pct_diff:.1f}%)<br>

        <p style="margin:5px 0; padding:5px; background:#f0f0f0; border-radius:3px;">
            {'<span style="color:#d00;">Position daha yüksek</span>' if diff > 0 else
             '<span style="color:#00d;">Connectivity daha yüksek</span>' if diff < 0 else
             '<span style="color:#0c0;">Benzer</span>'}
        </p>

        <b>Grid:</b> ({grid_lat:.4f}, {grid_lon:.4f})<br>
        <b>Stop Count:</b> {row['stop_count_conn']:.1f}
    </div>
    """

    folium.Rectangle(
        bounds=bounds,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.6,
        weight=1,
        popup=folium.Popup(popup_html, max_width=350),
        tooltip=f"Fark: {diff:.1f} ({pct_diff:.0f}%)"
    ).add_to(m)

# Legend ekle
legend_html = f"""
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 280px; height: auto;
            background-color: white; z-index:9999; font-size:12px;
            border:2px solid grey; border-radius: 5px; padding: 12px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">

    <h4 style="margin:0 0 10px 0;">Dağıtım Yöntemi Farkı</h4>

    <p style="margin:5px 0; font-size:11px;">
        <b>Connectivity-based</b> vs <b>Position-based</b><br>
        Ortalama fark: {df_diff['diff_yolcu'].mean():.2f} yolcu/durak
    </p>

    <div style="margin:10px 0;">
        <div style="display:flex; align-items:center; margin:3px 0;">
            <div style="width:20px; height:20px; background:#0066FF; margin-right:8px;"></div>
            <span>Connectivity daha yüksek (&lt;-10%)</span>
        </div>

        <div style="display:flex; align-items:center; margin:3px 0;">
            <div style="width:20px; height:20px; background:#00CC00; margin-right:8px;"></div>
            <span>Benzer (±10%)</span>
        </div>

        <div style="display:flex; align-items:center; margin:3px 0;">
            <div style="width:20px; height:20px; background:#FF6600; margin-right:8px;"></div>
            <span>Position daha yüksek (&gt;+10%)</span>
        </div>
    </div>

    <p style="margin:8px 0 0 0; font-size:10px; color:#666;">
        Toplam grid: {len(df_diff)}<br>
        Pozitif fark: {(df_diff['diff_yolcu'] > 0).sum()}<br>
        Negatif fark: {(df_diff['diff_yolcu'] < 0).sum()}<br>
        Benzer (±1): {((df_diff['diff_yolcu'] >= -1) & (df_diff['diff_yolcu'] <= 1)).sum()}
    </p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Başlık ekle
title_html = f"""
<div style="position: fixed;
            top: 10px; left: 50px; width: 600px; height: auto;
            background-color: white; z-index:9999;
            border:2px solid grey; border-radius: 5px; padding: 12px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
    <h3 style="margin:0; color: #333;">
        Yolcu Dağıtım Yöntemleri Karşılaştırma Haritası
    </h3>
    <p style="margin:5px 0; font-size:12px; color: #666;">
        Position-based vs Connectivity-based: Durak başına yolcu sayısı farkları
    </p>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# Kaydet
output_file = OUTPUT_DIR / 'distribution_difference_map.html'
m.save(str(output_file))
print(f"   [SAVED] {output_file}")

# ============================================================================
# 5. İSTATİSTİKLER
# ============================================================================
print("\n5. Difference statistics:")
print(f"   Grids where Position > Connectivity: {(df_diff['diff_yolcu'] > 0).sum()} ({(df_diff['diff_yolcu'] > 0).sum() / len(df_diff) * 100:.1f}%)")
print(f"   Grids where Connectivity > Position: {(df_diff['diff_yolcu'] < 0).sum()} ({(df_diff['diff_yolcu'] < 0).sum() / len(df_diff) * 100:.1f}%)")
print(f"   Grids with similar values (±1): {((df_diff['diff_yolcu'] >= -1) & (df_diff['diff_yolcu'] <= 1)).sum()}")

print("\n" + "=" * 70)
print("FARK HARİTASI OLUŞTURULDU!")
print(f"Harita: {output_file}")
print("=" * 70)
