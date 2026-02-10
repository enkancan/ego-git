"""
Create interactive HTML grid map for bus occupancy rate
Only includes bus routes (otobüs), excludes metro
"""

import pandas as pd
import numpy as np
import folium
from folium import Rectangle
import plotly.express as px
import plotly.graph_objects as go

print("=" * 70)
print("CREATING BUS OCCUPANCY GRID MAP")
print("=" * 70)

# 1. Load normalized data
print("\n1. Loading normalized transit data...")
df_norm = pd.read_csv('data/normalized_all_transit.csv', encoding='utf-8-sig')
print(f"   Total lines: {len(df_norm)}")

# Filter to only bus routes (exclude Metro)
df_norm_bus = df_norm[df_norm['mod'] != 'Metro'].copy()
print(f"   Bus lines only: {len(df_norm_bus)}")

# 2. Load stop coordinates
print("\n2. Loading stop coordinates...")
df_coords = pd.read_csv('data/ego_route_stops_all_coords.csv', encoding='utf-8-sig')

# Filter to Ankara
ANKARA_BOUNDS = {'lat': (39.5, 40.3), 'lon': (32.3, 33.2)}
df_coords = df_coords[
    (df_coords['final_latitude'] >= ANKARA_BOUNDS['lat'][0]) &
    (df_coords['final_latitude'] <= ANKARA_BOUNDS['lat'][1]) &
    (df_coords['final_longitude'] >= ANKARA_BOUNDS['lon'][0]) &
    (df_coords['final_longitude'] <= ANKARA_BOUNDS['lon'][1])
].copy()

print(f"   Stops in Ankara: {len(df_coords):,}")

# 3. Merge coordinates with bus data only
df_coords['hat_adi'] = df_coords['hat_adi'].astype(str)

df_merged = df_coords.merge(
    df_norm_bus[['hat_no', 'avg_yolcu', 'avg_sefer', 'avg_doluluk']],
    left_on='hat_adi',
    right_on='hat_no',
    how='inner'  # Only keep stops that match bus routes
)

print(f"   Merged bus stops: {len(df_merged):,}")
print(f"   Average occupancy: {df_merged['avg_doluluk'].mean():.1f}%")

# FIX: Calculate per-stop metrics to avoid duplication
df_merged['hat_stop_count'] = df_merged.groupby('hat_no')['durak_kodu'].transform('count')
df_merged['yolcu_per_stop'] = df_merged['avg_yolcu'] / df_merged['hat_stop_count']
df_merged['sefer_per_stop'] = df_merged['avg_sefer'] / df_merged['hat_stop_count']

# 4. Create 1km grid
print("\n3. Creating 1km grid for bus occupancy...")

GRID_SIZE_METERS = 1000
METERS_PER_DEGREE_LAT = 111000
METERS_PER_DEGREE_LON = 85000

GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON

df_merged['grid_lat'] = np.floor(df_merged['final_latitude'] / GRID_SIZE_LAT) * GRID_SIZE_LAT
df_merged['grid_lon'] = np.floor(df_merged['final_longitude'] / GRID_SIZE_LON) * GRID_SIZE_LON

# Aggregate by grid cell - use MEAN of per-stop metrics
df_grid = df_merged.groupby(['grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',
    'yolcu_per_stop': 'mean',   # Mean passengers per stop
    'sefer_per_stop': 'mean',   # Mean trips per stop
    'avg_doluluk': 'mean'       # Average occupancy rate
}).reset_index()

df_grid.columns = ['lat', 'lon', 'stop_count', 'avg_passengers_per_stop', 'avg_trips_per_stop', 'avg_occupancy']

# Calculate totals for the grid (for backwards compatibility)
df_grid['total_passengers'] = df_grid['avg_passengers_per_stop'] * df_grid['stop_count']
df_grid['total_trips'] = df_grid['avg_trips_per_stop'] * df_grid['stop_count']

print(f"   Grid cells: {len(df_grid):,}")
print(f"   Mean occupancy: {df_grid['avg_occupancy'].mean():.1f}%")
print(f"   Max occupancy: {df_grid['avg_occupancy'].max():.1f}%")
print(f"   Min occupancy: {df_grid['avg_occupancy'].min():.1f}%")

# Normalize occupancy
occ_mean = df_grid['avg_occupancy'].mean()
occ_std = df_grid['avg_occupancy'].std()
df_grid['normalized_occupancy'] = (df_grid['avg_occupancy'] - occ_mean) / occ_std

print(f"   Std occupancy: {occ_std:.1f}%")

# Color function for occupancy (different color scheme - green to red)
def get_color_occupancy(occupancy):
    """Get color based on occupancy percentage"""
    if occupancy < 40:
        return '#2166ac'  # Blue - very low
    elif occupancy < 50:
        return '#4393c3'  # Light blue - low
    elif occupancy < 60:
        return '#92c5de'  # Very light blue - below average
    elif occupancy < 70:
        return '#fddbc7'  # Very light orange - average
    elif occupancy < 80:
        return '#f4a582'  # Light orange - above average
    elif occupancy < 90:
        return '#d6604d'  # Orange - high
    else:
        return '#b2182b'  # Red - very high/overcrowded

# 5. Create occupancy map
print("\n4. Creating bus occupancy grid map...")
ankara_center = [39.93, 32.86]
m_occ = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

for _, cell in df_grid.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_occupancy(cell['avg_occupancy'])

    # Determine status
    if cell['avg_occupancy'] >= 85:
        status = 'COK YOGUN'
    elif cell['avg_occupancy'] >= 70:
        status = 'YOGUN'
    elif cell['avg_occupancy'] >= 50:
        status = 'NORMAL'
    else:
        status = 'AZ YOGUN'

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=0.5,
        popup=f"""
        <b>1km Grid - Otobus Doluluk Orani</b><br>
        Durak: {cell['stop_count']}<br>
        Ortalama Doluluk: {cell['avg_occupancy']:.1f}%<br>
        Durum: {status}<br>
        Ort. Yolcu/Durak: {cell['avg_passengers_per_stop']:,.1f}<br>
        Ort. Sefer/Durak: {cell['avg_trips_per_stop']:,.1f}<br>
        Toplam Tahmini Yolcu: {cell['total_passengers']:,.0f}<br>
        Toplam Tahmini Sefer: {cell['total_trips']:,.0f}
        """
    ).add_to(m_occ)

# Add legend
legend_html = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 260px; height: 320px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Otobus Doluluk Orani</b></p>
<p><i style="background:#b2182b;width:30px;height:20px;display:inline-block;"></i> Cok Yogun (>90%)</p>
<p><i style="background:#d6604d;width:30px;height:20px;display:inline-block;"></i> Yogun (80-90%)</p>
<p><i style="background:#f4a582;width:30px;height:20px;display:inline-block;"></i> Ortanin Ustu (70-80%)</p>
<p><i style="background:#fddbc7;width:30px;height:20px;display:inline-block;"></i> Normal (60-70%)</p>
<p><i style="background:#92c5de;width:30px;height:20px;display:inline-block;"></i> Az (50-60%)</p>
<p><i style="background:#4393c3;width:30px;height:20px;display:inline-block;"></i> Cok Az (40-50%)</p>
<p><i style="background:#2166ac;width:30px;height:20px;display:inline-block;"></i> Cok Dusuk (<40%)</p>
<p style="margin-top:10px;"><small>Sadece Otobus Hatlari<br>(Metro Dahil Degil)</small></p>
</div>
'''
m_occ.get_root().html.add_child(folium.Element(legend_html))

m_occ.save('maps/interactive_grid_bus_occupancy.html')
print("   [SAVED] maps/interactive_grid_bus_occupancy.html")

# 6. Create Plotly charts
print("\n5. Creating occupancy analysis charts...")

# Chart 1: Top 20 most crowded cells
top20_crowded = df_grid.nlargest(20, 'avg_occupancy').copy()
top20_crowded['location'] = top20_crowded.apply(lambda r: f"({r['lat']:.3f}, {r['lon']:.3f})", axis=1)
top20_crowded['status'] = top20_crowded['avg_occupancy'].apply(
    lambda x: 'Cok Yogun' if x >= 85 else 'Yogun' if x >= 70 else 'Normal'
)

fig1 = px.bar(top20_crowded,
              x='avg_occupancy',
              y='location',
              color='status',
              color_discrete_map={'Cok Yogun': '#b2182b', 'Yogun': '#d6604d', 'Normal': '#f4a582'},
              orientation='h',
              title='En Yoğun 20 Grid Hücresi (Doluluk Oranına Göre - Sadece Otobüs)',
              labels={'avg_occupancy': 'Ortalama Doluluk %', 'location': 'Grid Koordinat', 'status': 'Durum'})

fig1.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Yoğun Eşik (70%)")
fig1.add_vline(x=85, line_dash="dash", line_color="red", annotation_text="Çok Yoğun (85%)")
fig1.update_layout(height=700)
fig1.write_html('maps/interactive_top20_bus_occupancy.html')
print("   [SAVED] maps/interactive_top20_bus_occupancy.html")

# Chart 2: Occupancy distribution histogram
fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=df_grid['avg_occupancy'],
    name='Doluluk Orani',
    marker_color='steelblue',
    nbinsx=30
))

fig2.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Yogun Esik (70%)")
fig2.add_vline(x=occ_mean, line_dash="solid", line_color="green", annotation_text=f"Ortalama ({occ_mean:.1f}%)")

fig2.update_layout(
    title='Otobüs Doluluk Oranı Dağılımı (Grid Hücreleri)',
    xaxis_title='Ortalama Doluluk Oranı (%)',
    yaxis_title='Grid Hücre Sayısı',
    height=600
)

fig2.write_html('maps/interactive_bus_occupancy_distribution.html')
print("   [SAVED] maps/interactive_bus_occupancy_distribution.html")

# Chart 3: Scatter - Passengers vs Occupancy
fig3 = px.scatter(df_grid,
                  x='total_passengers',
                  y='avg_occupancy',
                  size='stop_count',
                  color='avg_occupancy',
                  color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green reversed
                  hover_data=['total_trips', 'stop_count'],
                  title='Yolcu Sayısı vs Doluluk Oranı (Otobüs Grid Hücreleri)',
                  labels={'total_passengers': 'Toplam Yolcu/Gün',
                          'avg_occupancy': 'Ortalama Doluluk %',
                          'stop_count': 'Durak Sayısı'})

fig3.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Yogun Esik (70%)")
fig3.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Cok Yogun (85%)")
fig3.update_layout(height=700)
fig3.write_html('maps/interactive_scatter_bus_occupancy.html')
print("   [SAVED] maps/interactive_scatter_bus_occupancy.html")

# 7. Create dashboard
print("\n6. Creating bus occupancy dashboard...")

# Find problematic areas (high occupancy)
high_occ = df_grid[df_grid['avg_occupancy'] >= 85].sort_values('avg_occupancy', ascending=False)
medium_occ = df_grid[(df_grid['avg_occupancy'] >= 70) & (df_grid['avg_occupancy'] < 85)].sort_values('avg_occupancy', ascending=False)

dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EGO Otobüs - Doluluk Oranı Analizi</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 180px; margin: 10px; }}
        .stat-number {{ font-size: 36px; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        .links {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .links h2 {{ color: #2c3e50; }}
        .links a {{ display: block; padding: 10px; margin: 5px 0; background: #3498db; color: white; text-decoration: none; border-radius: 4px; }}
        .links a:hover {{ background: #2980b9; }}
        .warning {{ background: #e74c3c; color: white; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        .info {{ background: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #3498db; }}
        .red {{ color: #e74c3c; }}
        .orange {{ color: #e67e22; }}
        .green {{ color: #27ae60; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚌 EGO Otobüs - Doluluk Oranı Analizi</h1>
        <p>1km x 1km Grid Bazlı - Sadece Otobüs Hatları (Metro Hariç)</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{len(df_grid):,}</div>
            <div class="stat-label">Grid Hücresi</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{occ_mean:.1f}%</div>
            <div class="stat-label">Ortalama Doluluk</div>
        </div>
        <div class="stat-box">
            <div class="stat-number red">{len(high_occ)}</div>
            <div class="stat-label">Çok Yoğun (>85%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-number orange">{len(medium_occ)}</div>
            <div class="stat-label">Yoğun (70-85%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-number green">{len(df_grid[df_grid['avg_occupancy'] < 70])}</div>
            <div class="stat-label">Normal (<70%)</div>
        </div>
    </div>

    <div class="links">
        <h2>🗺️ İnteraktif Haritalar</h2>
        <a href="interactive_grid_bus_occupancy.html" target="_blank">
            🚌 Otobüs Doluluk Oranı Haritası (1km Grid)
        </a>
    </div>

    <div class="links">
        <h2>📊 İnteraktif Analizler</h2>
        <a href="interactive_top20_bus_occupancy.html" target="_blank">
            📊 En Yoğun 20 Grid Hücresi
        </a>
        <a href="interactive_bus_occupancy_distribution.html" target="_blank">
            📉 Doluluk Oranı Dağılımı
        </a>
        <a href="interactive_scatter_bus_occupancy.html" target="_blank">
            📈 Yolcu Sayısı vs Doluluk Oranı
        </a>
    </div>

    <div class="warning">
        <h3>⚠️ Kritik Yoğun Bölgeler (Doluluk >85%)</h3>
        <p>Bu bölgelerde otobüsler aşırı dolu. Ek araç ya da sefer gerekebilir:</p>
"""

if len(high_occ) > 0:
    for idx, row in high_occ.head(10).iterrows():
        dashboard_html += f'        <p><strong>•</strong> ({row["lat"]:.3f}, {row["lon"]:.3f}) - <strong>{row["avg_occupancy"]:.1f}%</strong> doluluk, {row["stop_count"]} durak, {row["total_passengers"]:,.0f} yolcu/gün</p>\n'
else:
    dashboard_html += '        <p>Kritik yoğun bölge yok!</p>\n'

dashboard_html += """
    </div>

    <div class="info">
        <h3>ℹ️ Doluluk Oranı Açıklaması</h3>
        <p><strong>Doluluk Oranı:</strong> Otobüsteki yolcu sayısının araç kapasitesine oranı</p>
        <p><strong><70%:</strong> Normal doluluk - rahat yolculuk</p>
        <p><strong>70-85%:</strong> Yoğun - yolcular ayakta olabilir</p>
        <p><strong>>85%:</strong> Çok yoğun - kapasite aşımı riski</p>
        <p><strong>Grid Boyutu:</strong> 1km x 1km</p>
        <p><strong>Not:</strong> Bu analiz sadece otobüs hatlarını içerir, metro dahil değildir</p>
    </div>

    <div class="links">
        <h2>🏆 En Yoğun 15 Grid Hücresi</h2>
"""

for idx, row in df_grid.nlargest(15, 'avg_occupancy').iterrows():
    status = 'Cok Yogun' if row['avg_occupancy'] >= 85 else 'Yogun' if row['avg_occupancy'] >= 70 else 'Normal'
    color_class = 'red' if row['avg_occupancy'] >= 85 else 'orange' if row['avg_occupancy'] >= 70 else 'green'
    dashboard_html += f'        <p><strong>{idx+1}.</strong> ({row["lat"]:.3f}, {row["lon"]:.3f}) - <span class="{color_class}"><strong>{row["avg_occupancy"]:.1f}%</strong></span> ({status}), {row["stop_count"]} durak</p>\n'

dashboard_html += """
    </div>
</body>
</html>
"""

with open('maps/bus_occupancy_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print("   [SAVED] maps/bus_occupancy_dashboard.html")

# 8. Summary stats
print(f"\n{'=' * 70}")
print("BUS OCCUPANCY ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOccupancy Statistics:")
print(f"  Mean: {occ_mean:.1f}%")
print(f"  Median: {df_grid['avg_occupancy'].median():.1f}%")
print(f"  Std: {occ_std:.1f}%")
print(f"  Max: {df_grid['avg_occupancy'].max():.1f}%")
print(f"  Min: {df_grid['avg_occupancy'].min():.1f}%")

print(f"\nGrid Distribution:")
print(f"  Very High (>85%): {len(high_occ)} cells")
print(f"  High (70-85%): {len(medium_occ)} cells")
print(f"  Normal (<70%): {len(df_grid[df_grid['avg_occupancy'] < 70])} cells")

print("\nCreated files:")
print("  1. maps/bus_occupancy_dashboard.html - Ana dashboard")
print("  2. maps/interactive_grid_bus_occupancy.html - Grid harita")
print("  3. maps/interactive_top20_bus_occupancy.html - Top 20 yogun")
print("  4. maps/interactive_bus_occupancy_distribution.html - Dagilim")
print("  5. maps/interactive_scatter_bus_occupancy.html - Yolcu vs doluluk")
print("\nTarayicida bus_occupancy_dashboard.html dosyasini ac!")
