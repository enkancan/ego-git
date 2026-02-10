"""
Create interactive HTML grid maps for different transit metrics:
1. Sefer Sayısı (Trip Count)
2. Araç Kapasitesi (Vehicle Capacity)
"""

import pandas as pd
import numpy as np
import folium
from folium import Rectangle
import plotly.express as px
import plotly.graph_objects as go

print("=" * 70)
print("CREATING GRID MAPS BY DIFFERENT METRICS")
print("=" * 70)

# 1. Load normalized data
print("\n1. Loading normalized transit data...")
df_norm = pd.read_csv('data/normalized_all_transit.csv', encoding='utf-8-sig')
print(f"   Total lines: {len(df_norm)}")

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

# 3. Prepare metro route matching
# Metro routes: A1-D/A1-G -> A1, M1-D/M1-G -> M1-M2-M3, M4-D/M4-G -> M4, T1-D/T1-G -> T1
df_coords['hat_adi'] = df_coords['hat_adi'].astype(str)
df_coords['hat_base'] = df_coords['hat_adi'].str.replace('-D', '', regex=False).str.replace('-G', '', regex=False)

# Expand M1-M2-M3 to match M1, M2, M3
df_norm_exp = df_norm.copy()
m123 = df_norm_exp[df_norm_exp['hat_no'] == 'M1-M2-M3']
if len(m123) > 0:
    for code in ['M1', 'M2', 'M3']:
        row = m123.copy()
        row['hat_no'] = code
        df_norm_exp = pd.concat([df_norm_exp, row], ignore_index=True)

# Fix T1 encoding issues
df_norm_exp['hat_no'] = df_norm_exp['hat_no'].replace('T1TELEFERIK', 'T1')
df_norm_exp['hat_no'] = df_norm_exp['hat_no'].str.replace('T1TELEFER.*', 'T1', regex=True)

# Merge
df_merged = df_coords.merge(
    df_norm_exp[['hat_no', 'avg_yolcu', 'avg_sefer', 'avg_doluluk', 'mod']],
    left_on='hat_base',
    right_on='hat_no',
    how='left'
)

# Fill missing
df_merged['avg_yolcu'] = df_merged['avg_yolcu'].fillna(0)
df_merged['avg_sefer'] = df_merged['avg_sefer'].fillna(0)
df_merged['avg_doluluk'] = df_merged['avg_doluluk'].fillna(0)
df_merged['mod'] = df_merged['mod'].fillna('Otobus')

print(f"   Merged records: {len(df_merged):,}")
print(f"   Metro stops: {len(df_merged[df_merged['mod']=='Metro']):,}")
print(f"   Bus stops: {len(df_merged[df_merged['mod']=='Otobus']):,}")

# FIX: Calculate per-stop metrics to avoid duplication
df_merged['hat_stop_count'] = df_merged.groupby('hat_no')['durak_kodu'].transform('count')
df_merged['yolcu_per_stop'] = df_merged['avg_yolcu'] / df_merged['hat_stop_count']
df_merged['sefer_per_stop'] = df_merged['avg_sefer'] / df_merged['hat_stop_count']

# 4. Create 1km grid
print("\n3. Creating 1km grid...")

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
    'yolcu_per_stop': 'mean',    # Mean passengers per stop
    'sefer_per_stop': 'mean',    # Mean trips per stop
    'avg_doluluk': 'mean',       # Average occupancy
    'mod': lambda x: 'Metro+Bus' if len(set(x)) > 1 else x.iloc[0]
}).reset_index()

df_grid.columns = ['lat', 'lon', 'stop_count', 'avg_passengers_per_stop', 'avg_trips_per_stop', 'avg_occupancy', 'primary_mode']

# Calculate totals for the grid (for backwards compatibility)
df_grid['total_passengers'] = df_grid['avg_passengers_per_stop'] * df_grid['stop_count']
df_grid['total_trips'] = df_grid['avg_trips_per_stop'] * df_grid['stop_count']

print(f"   Grid cells: {len(df_grid):,}")
print(f"   Max trips per cell: {df_grid['total_trips'].max():,.0f}")

# Normalize trip counts
trip_mean = df_grid['total_trips'].mean()
trip_std = df_grid['total_trips'].std()
df_grid['normalized_trips'] = (df_grid['total_trips'] - trip_mean) / trip_std

print(f"   Mean trips per cell: {trip_mean:,.0f}")
print(f"   Std trips: {trip_std:,.0f}")

# 5. Calculate vehicle capacity per grid (estimate based on trips and occupancy)
# Capacity = passengers / occupancy_rate
df_grid['estimated_capacity'] = df_grid.apply(
    lambda row: row['total_passengers'] / (row['avg_occupancy'] / 100) if row['avg_occupancy'] > 0 else 0,
    axis=1
)

# Normalize capacity
cap_mean = df_grid['estimated_capacity'].mean()
cap_std = df_grid['estimated_capacity'].std()
df_grid['normalized_capacity'] = (df_grid['estimated_capacity'] - cap_mean) / cap_std

print(f"   Mean capacity per cell: {cap_mean:,.0f}")
print(f"   Std capacity: {cap_std:,.0f}")

# Color function for normalized values
def get_color_normalized(normalized_val):
    """Get color based on normalized value (z-score)"""
    if normalized_val < -0.5:
        return '#fee5d9'  # Very light (below average)
    elif normalized_val < 0:
        return '#fcae91'  # Light (slightly below average)
    elif normalized_val < 0.5:
        return '#fb6a4a'  # Medium (slightly above average)
    elif normalized_val < 1.0:
        return '#de2d26'  # Dark (above average)
    elif normalized_val < 2.0:
        return '#a50f15'  # Very dark (well above average)
    else:
        return '#67000d'  # Darkest (extremely high)

# 6. Create map for TRIP COUNT
print("\n4. Creating trip count grid map...")
ankara_center = [39.93, 32.86]
m_trips = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

for _, cell in df_grid.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_normalized(cell['normalized_trips'])

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=0.5,
        popup=f"""
        <b>1km Grid - Sefer Sayisi</b><br>
        Durak: {cell['stop_count']}<br>
        Ort. Sefer/Durak: {cell['avg_trips_per_stop']:,.1f}<br>
        Toplam Tahmini Sefer: {cell['total_trips']:,.0f}<br>
        Normalize Skor: {cell['normalized_trips']:.2f}<br>
        Mod: {cell['primary_mode']}
        """
    ).add_to(m_trips)

# Add legend
legend_html = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 250px; height: 280px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Sefer Sayisi Yogunlugu</b></p>
<p><i style="background:#67000d;width:30px;height:20px;display:inline-block;"></i> Cok Yuksek (>2σ)</p>
<p><i style="background:#a50f15;width:30px;height:20px;display:inline-block;"></i> Yuksek (1-2σ)</p>
<p><i style="background:#de2d26;width:30px;height:20px;display:inline-block;"></i> Ortanin Ustu (0.5-1σ)</p>
<p><i style="background:#fb6a4a;width:30px;height:20px;display:inline-block;"></i> Orta (-0.5-0.5σ)</p>
<p><i style="background:#fcae91;width:30px;height:20px;display:inline-block;"></i> Ortanin Alti (-0.5-0σ)</p>
<p><i style="background:#fee5d9;width:30px;height:20px;display:inline-block;"></i> Dusuk (<-0.5σ)</p>
<p style="margin-top:10px;"><small>σ = Standart Sapma<br>0 = Ortalama Yogunluk</small></p>
</div>
'''
m_trips.get_root().html.add_child(folium.Element(legend_html))

m_trips.save('maps/interactive_grid_trip_count.html')
print("   [SAVED] maps/interactive_grid_trip_count.html")

# 7. Create map for VEHICLE CAPACITY
print("\n5. Creating vehicle capacity grid map...")
m_capacity = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

for _, cell in df_grid.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_normalized(cell['normalized_capacity'])

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=0.5,
        popup=f"""
        <b>1km Grid - Arac Kapasitesi</b><br>
        Durak: {cell['stop_count']}<br>
        Ort. Yolcu/Durak: {cell['avg_passengers_per_stop']:,.1f}<br>
        Tahmini Toplam Kapasite: {cell['estimated_capacity']:,.0f}<br>
        Normalize Skor: {cell['normalized_capacity']:.2f}<br>
        Ortalama Doluluk: {cell['avg_occupancy']:.1f}%<br>
        Mod: {cell['primary_mode']}
        """
    ).add_to(m_capacity)

# Add legend
legend_html_cap = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 250px; height: 280px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Arac Kapasitesi Yogunlugu</b></p>
<p><i style="background:#67000d;width:30px;height:20px;display:inline-block;"></i> Cok Yuksek (>2σ)</p>
<p><i style="background:#a50f15;width:30px;height:20px;display:inline-block;"></i> Yuksek (1-2σ)</p>
<p><i style="background:#de2d26;width:30px;height:20px;display:inline-block;"></i> Ortanin Ustu (0.5-1σ)</p>
<p><i style="background:#fb6a4a;width:30px;height:20px;display:inline-block;"></i> Orta (-0.5-0.5σ)</p>
<p><i style="background:#fcae91;width:30px;height:20px;display:inline-block;"></i> Ortanin Alti (-0.5-0σ)</p>
<p><i style="background:#fee5d9;width:30px;height:20px;display:inline-block;"></i> Dusuk (<-0.5σ)</p>
<p style="margin-top:10px;"><small>σ = Standart Sapma<br>0 = Ortalama Kapasite</small></p>
</div>
'''
m_capacity.get_root().html.add_child(folium.Element(legend_html_cap))

m_capacity.save('maps/interactive_grid_capacity.html')
print("   [SAVED] maps/interactive_grid_capacity.html")

# 8. Create Plotly comparison charts
print("\n6. Creating comparison charts...")

# Chart 1: Top 20 cells by trips
top20_trips = df_grid.nlargest(20, 'total_trips').copy()
top20_trips['location'] = top20_trips.apply(lambda r: f"({r['lat']:.3f}, {r['lon']:.3f})", axis=1)

fig1 = px.bar(top20_trips,
              x='total_trips',
              y='location',
              color='primary_mode',
              color_discrete_map={'Metro': 'red', 'Otobüs': 'steelblue', 'Metro+Bus': 'purple'},
              orientation='h',
              title='En Yoğun 20 Grid Hücresi (Sefer Sayısına Göre)',
              labels={'total_trips': 'Toplam Sefer/Gün', 'location': 'Grid Koordinat', 'primary_mode': 'Mod'})

fig1.update_layout(height=600)
fig1.write_html('maps/interactive_top20_trips.html')
print("   [SAVED] maps/interactive_top20_trips.html")

# Chart 2: Top 20 cells by capacity
top20_cap = df_grid.nlargest(20, 'estimated_capacity').copy()
top20_cap['location'] = top20_cap.apply(lambda r: f"({r['lat']:.3f}, {r['lon']:.3f})", axis=1)

fig2 = px.bar(top20_cap,
              x='estimated_capacity',
              y='location',
              color='primary_mode',
              color_discrete_map={'Metro': 'red', 'Otobüs': 'steelblue', 'Metro+Bus': 'purple'},
              orientation='h',
              title='En Yüksek 20 Grid Hücresi (Araç Kapasitesine Göre)',
              labels={'estimated_capacity': 'Tahmini Kapasite', 'location': 'Grid Koordinat', 'primary_mode': 'Mod'})

fig2.update_layout(height=600)
fig2.write_html('maps/interactive_top20_capacity.html')
print("   [SAVED] maps/interactive_top20_capacity.html")

# Chart 3: Scatter - Trips vs Capacity
fig3 = px.scatter(df_grid,
                  x='total_trips',
                  y='estimated_capacity',
                  color='primary_mode',
                  size='stop_count',
                  color_discrete_map={'Metro': 'red', 'Otobüs': 'steelblue', 'Metro+Bus': 'purple'},
                  hover_data=['stop_count', 'avg_occupancy'],
                  title='Sefer Sayısı vs Araç Kapasitesi (Grid Hücreleri)',
                  labels={'total_trips': 'Toplam Sefer/Gün',
                          'estimated_capacity': 'Tahmini Kapasite',
                          'stop_count': 'Durak Sayısı',
                          'primary_mode': 'Mod'})

fig3.update_layout(height=700)
fig3.write_html('maps/interactive_scatter_trips_capacity.html')
print("   [SAVED] maps/interactive_scatter_trips_capacity.html")

# 9. Create summary dashboard
print("\n7. Creating metrics dashboard...")

dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EGO Ulaşım - Sefer ve Kapasite Analizi</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 200px; margin: 10px; }}
        .stat-number {{ font-size: 36px; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        .links {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .links h2 {{ color: #2c3e50; }}
        .links a {{ display: block; padding: 10px; margin: 5px 0; background: #3498db; color: white; text-decoration: none; border-radius: 4px; }}
        .links a:hover {{ background: #2980b9; }}
        .info {{ background: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚌 EGO Ulaşım - Sefer Sayısı ve Araç Kapasitesi Analizi</h1>
        <p>1km x 1km Grid Bazlı İnteraktif Görselleştirmeler</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{len(df_grid):,}</div>
            <div class="stat-label">Grid Hücresi</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{df_grid['total_trips'].sum():,.0f}</div>
            <div class="stat-label">Toplam Sefer/Gün</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{trip_mean:,.0f}</div>
            <div class="stat-label">Ortalama Sefer/Grid</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{cap_mean:,.0f}</div>
            <div class="stat-label">Ortalama Kapasite/Grid</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{df_grid['stop_count'].sum():,}</div>
            <div class="stat-label">Toplam Durak</div>
        </div>
    </div>

    <div class="links">
        <h2>🗺️ İnteraktif Grid Haritaları</h2>
        <a href="interactive_grid_trip_count.html" target="_blank">
            🚌 Sefer Sayısı Yoğunluk Haritası (1km Grid)
        </a>
        <a href="interactive_grid_capacity.html" target="_blank">
            🚍 Araç Kapasitesi Yoğunluk Haritası (1km Grid)
        </a>
        <a href="interactive_transit_density.html" target="_blank">
            👥 Yolcu Sayısı Yoğunluk Haritası (1km Grid)
        </a>
    </div>

    <div class="links">
        <h2>📊 İnteraktif Karşılaştırma Grafikleri</h2>
        <a href="interactive_top20_trips.html" target="_blank">
            📊 En Yoğun 20 Grid - Sefer Sayısı
        </a>
        <a href="interactive_top20_capacity.html" target="_blank">
            📊 En Yüksek 20 Grid - Araç Kapasitesi
        </a>
        <a href="interactive_scatter_trips_capacity.html" target="_blank">
            📈 Sefer vs Kapasite - Scatter Plot
        </a>
    </div>

    <div class="info">
        <h3>ℹ️ Metrik Açıklamaları</h3>
        <p><strong>Sefer Sayısı:</strong> Her grid hücresindeki tüm hatların günlük toplam sefer sayısı</p>
        <p><strong>Araç Kapasitesi:</strong> Yolcu sayısı ve doluluk oranından hesaplanan tahmini araç kapasitesi</p>
        <p><strong>Normalizasyon:</strong> Z-score kullanılarak tüm grid hücreleri normalize edilmiştir</p>
        <p><strong>Grid Boyutu:</strong> 1km x 1km (Ankara sınırları içinde)</p>
    </div>

    <div class="links">
        <h2>🏆 En Yoğun 10 Grid Hücresi (Sefer Sayısı)</h2>
"""

for idx, row in df_grid.nlargest(10, 'total_trips').iterrows():
    dashboard_html += f'        <p><strong>{idx+1}.</strong> ({row["lat"]:.3f}, {row["lon"]:.3f}) - {row["total_trips"]:,.0f} sefer/gün, {row["stop_count"]} durak ({row["primary_mode"]})</p>\n'

dashboard_html += """
    </div>

    <div class="links">
        <h2>🏆 En Yüksek 10 Grid Hücresi (Kapasite)</h2>
"""

for idx, row in df_grid.nlargest(10, 'estimated_capacity').iterrows():
    dashboard_html += f'        <p><strong>{idx+1}.</strong> ({row["lat"]:.3f}, {row["lon"]:.3f}) - {row["estimated_capacity"]:,.0f} kapasite, {row["avg_occupancy"]:.1f}% doluluk ({row["primary_mode"]})</p>\n'

dashboard_html += """
    </div>
</body>
</html>
"""

with open('maps/metrics_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print("   [SAVED] maps/metrics_dashboard.html")

print(f"\n{'=' * 70}")
print("METRIC GRID MAPS COMPLETE")
print("=" * 70)
print("\nCreated files:")
print("  1. maps/metrics_dashboard.html - Ana dashboard (buradan başla!)")
print("  2. maps/interactive_grid_trip_count.html - Sefer sayısı grid harita")
print("  3. maps/interactive_grid_capacity.html - Kapasite grid harita")
print("  4. maps/interactive_top20_trips.html - Top 20 sefer")
print("  5. maps/interactive_top20_capacity.html - Top 20 kapasite")
print("  6. maps/interactive_scatter_trips_capacity.html - Sefer vs kapasite")
print("\nTarayıcıda metrics_dashboard.html dosyasını aç!")
