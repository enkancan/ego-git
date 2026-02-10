"""
Create 4-map comparison dashboard for bus metrics
Shows same grid cells with different metrics side-by-side
"""

import pandas as pd
import numpy as np
import folium
from folium import Rectangle

print("=" * 70)
print("CREATING 4-MAP BUS COMPARISON DASHBOARD")
print("=" * 70)

# 1. Load normalized data
print("\n1. Loading data...")
df_norm = pd.read_csv('data/normalized_all_transit.csv', encoding='utf-8-sig')
df_coords = pd.read_csv('data/ego_route_stops_all_coords.csv', encoding='utf-8-sig')

# Filter to bus only
df_norm_bus = df_norm[df_norm['mod'] != 'Metro'].copy()
print(f"   Bus lines: {len(df_norm_bus)}")

# Filter Ankara bounds
ANKARA_BOUNDS = {'lat': (39.5, 40.3), 'lon': (32.3, 33.2)}
df_coords = df_coords[
    (df_coords['final_latitude'] >= ANKARA_BOUNDS['lat'][0]) &
    (df_coords['final_latitude'] <= ANKARA_BOUNDS['lat'][1]) &
    (df_coords['final_longitude'] >= ANKARA_BOUNDS['lon'][0]) &
    (df_coords['final_longitude'] <= ANKARA_BOUNDS['lon'][1])
].copy()

# Merge
df_coords['hat_adi'] = df_coords['hat_adi'].astype(str)
df_merged = df_coords.merge(
    df_norm_bus[['hat_no', 'avg_yolcu', 'avg_sefer', 'avg_doluluk']],
    left_on='hat_adi',
    right_on='hat_no',
    how='inner'
)

print(f"   Merged bus stops: {len(df_merged):,}")

# 2. Create 1km grid
print("\n2. Creating 1km grid...")
GRID_SIZE_METERS = 1000
METERS_PER_DEGREE_LAT = 111000
METERS_PER_DEGREE_LON = 85000
GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON

df_merged['grid_lat'] = np.floor(df_merged['final_latitude'] / GRID_SIZE_LAT) * GRID_SIZE_LAT
df_merged['grid_lon'] = np.floor(df_merged['final_longitude'] / GRID_SIZE_LON) * GRID_SIZE_LON

# Aggregate by grid
df_grid = df_merged.groupby(['grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',
    'avg_yolcu': 'sum',
    'avg_sefer': 'sum',
    'avg_doluluk': 'mean'
}).reset_index()

df_grid.columns = ['lat', 'lon', 'stop_count', 'total_passengers', 'total_trips', 'avg_occupancy']

# Calculate capacity (estimate)
df_grid['estimated_capacity'] = df_grid.apply(
    lambda row: row['total_passengers'] / (row['avg_occupancy'] / 100) if row['avg_occupancy'] > 0 else 0,
    axis=1
)

print(f"   Grid cells: {len(df_grid):,}")

# Normalize each metric for color scaling
for col in ['total_passengers', 'total_trips', 'estimated_capacity', 'avg_occupancy']:
    mean = df_grid[col].mean()
    std = df_grid[col].std()
    df_grid[f'{col}_norm'] = (df_grid[col] - mean) / std

print(f"   Metrics normalized")

# 3. Color functions for each metric
def get_color_passengers(norm_val):
    """Red scale for passengers"""
    if norm_val < -0.5: return '#fee5d9'
    elif norm_val < 0: return '#fcae91'
    elif norm_val < 0.5: return '#fb6a4a'
    elif norm_val < 1.0: return '#de2d26'
    elif norm_val < 2.0: return '#a50f15'
    else: return '#67000d'

def get_color_trips(norm_val):
    """Blue scale for trips"""
    if norm_val < -0.5: return '#eff3ff'
    elif norm_val < 0: return '#bdd7e7'
    elif norm_val < 0.5: return '#6baed6'
    elif norm_val < 1.0: return '#3182bd'
    elif norm_val < 2.0: return '#08519c'
    else: return '#08306b'

def get_color_capacity(norm_val):
    """Green scale for capacity"""
    if norm_val < -0.5: return '#edf8e9'
    elif norm_val < 0: return '#bae4b3'
    elif norm_val < 0.5: return '#74c476'
    elif norm_val < 1.0: return '#31a354'
    elif norm_val < 2.0: return '#006d2c'
    else: return '#00441b'

def get_color_occupancy(occupancy):
    """Orange-Red scale for occupancy"""
    if occupancy < 40: return '#2166ac'
    elif occupancy < 50: return '#4393c3'
    elif occupancy < 60: return '#92c5de'
    elif occupancy < 70: return '#fddbc7'
    elif occupancy < 80: return '#f4a582'
    elif occupancy < 90: return '#d6604d'
    else: return '#b2182b'

# 4. Create 4 maps
print("\n3. Creating 4 maps...")
ankara_center = [39.93, 32.86]

# Map 1: Passengers
print("   Map 1: Yolcu Yogunlugu...")
m1 = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')
for _, cell in df_grid.iterrows():
    bounds = [[cell['lat'], cell['lon']], [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]]
    color = get_color_passengers(cell['total_passengers_norm'])
    folium.Rectangle(
        bounds=bounds, color='black', fill=True, fillColor=color, fillOpacity=0.7, weight=0.5,
        popup=f"<b>Yolcu Yogunlugu</b><br>Yolcu: {cell['total_passengers']:,.0f}<br>Durak: {cell['stop_count']}"
    ).add_to(m1)

legend1 = '''<div style="position:fixed;top:10px;right:10px;width:200px;background:white;border:2px solid grey;z-index:9999;padding:10px">
<b>Yolcu Yogunlugu</b><br>
<i style="background:#67000d;width:20px;height:15px;display:inline-block"></i> Cok Yuksek<br>
<i style="background:#a50f15;width:20px;height:15px;display:inline-block"></i> Yuksek<br>
<i style="background:#de2d26;width:20px;height:15px;display:inline-block"></i> Orta-Ust<br>
<i style="background:#fb6a4a;width:20px;height:15px;display:inline-block"></i> Orta<br>
<i style="background:#fcae91;width:20px;height:15px;display:inline-block"></i> Dusuk<br>
</div>'''
m1.get_root().html.add_child(folium.Element(legend1))
m1.save('maps/temp_map1_passengers.html')

# Map 2: Trips
print("   Map 2: Sefer Sayisi...")
m2 = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')
for _, cell in df_grid.iterrows():
    bounds = [[cell['lat'], cell['lon']], [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]]
    color = get_color_trips(cell['total_trips_norm'])
    folium.Rectangle(
        bounds=bounds, color='black', fill=True, fillColor=color, fillOpacity=0.7, weight=0.5,
        popup=f"<b>Sefer Sayisi</b><br>Sefer: {cell['total_trips']:,.0f}<br>Durak: {cell['stop_count']}"
    ).add_to(m2)

legend2 = '''<div style="position:fixed;top:10px;right:10px;width:200px;background:white;border:2px solid grey;z-index:9999;padding:10px">
<b>Sefer Sayisi</b><br>
<i style="background:#08306b;width:20px;height:15px;display:inline-block"></i> Cok Yuksek<br>
<i style="background:#08519c;width:20px;height:15px;display:inline-block"></i> Yuksek<br>
<i style="background:#3182bd;width:20px;height:15px;display:inline-block"></i> Orta-Ust<br>
<i style="background:#6baed6;width:20px;height:15px;display:inline-block"></i> Orta<br>
<i style="background:#bdd7e7;width:20px;height:15px;display:inline-block"></i> Dusuk<br>
</div>'''
m2.get_root().html.add_child(folium.Element(legend2))
m2.save('maps/temp_map2_trips.html')

# Map 3: Capacity
print("   Map 3: Arac Kapasitesi...")
m3 = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')
for _, cell in df_grid.iterrows():
    bounds = [[cell['lat'], cell['lon']], [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]]
    color = get_color_capacity(cell['estimated_capacity_norm'])
    folium.Rectangle(
        bounds=bounds, color='black', fill=True, fillColor=color, fillOpacity=0.7, weight=0.5,
        popup=f"<b>Arac Kapasitesi</b><br>Kapasite: {cell['estimated_capacity']:,.0f}<br>Durak: {cell['stop_count']}"
    ).add_to(m3)

legend3 = '''<div style="position:fixed;top:10px;right:10px;width:200px;background:white;border:2px solid grey;z-index:9999;padding:10px">
<b>Arac Kapasitesi</b><br>
<i style="background:#00441b;width:20px;height:15px;display:inline-block"></i> Cok Yuksek<br>
<i style="background:#006d2c;width:20px;height:15px;display:inline-block"></i> Yuksek<br>
<i style="background:#31a354;width:20px;height:15px;display:inline-block"></i> Orta-Ust<br>
<i style="background:#74c476;width:20px;height:15px;display:inline-block"></i> Orta<br>
<i style="background:#bae4b3;width:20px;height:15px;display:inline-block"></i> Dusuk<br>
</div>'''
m3.get_root().html.add_child(folium.Element(legend3))
m3.save('maps/temp_map3_capacity.html')

# Map 4: Occupancy
print("   Map 4: Doluluk Orani...")
m4 = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')
for _, cell in df_grid.iterrows():
    bounds = [[cell['lat'], cell['lon']], [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]]
    color = get_color_occupancy(cell['avg_occupancy'])
    folium.Rectangle(
        bounds=bounds, color='black', fill=True, fillColor=color, fillOpacity=0.7, weight=0.5,
        popup=f"<b>Doluluk Orani</b><br>Doluluk: {cell['avg_occupancy']:.1f}%<br>Durak: {cell['stop_count']}"
    ).add_to(m4)

legend4 = '''<div style="position:fixed;top:10px;right:10px;width:200px;background:white;border:2px solid grey;z-index:9999;padding:10px">
<b>Doluluk Orani</b><br>
<i style="background:#b2182b;width:20px;height:15px;display:inline-block"></i> >90%<br>
<i style="background:#d6604d;width:20px;height:15px;display:inline-block"></i> 80-90%<br>
<i style="background:#f4a582;width:20px;height:15px;display:inline-block"></i> 70-80%<br>
<i style="background:#fddbc7;width:20px;height:15px;display:inline-block"></i> 60-70%<br>
<i style="background:#92c5de;width:20px;height:15px;display:inline-block"></i> <60%<br>
</div>'''
m4.get_root().html.add_child(folium.Element(legend4))
m4.save('maps/temp_map4_occupancy.html')

# 5. Create combined HTML with 4 maps
print("\n4. Creating combined dashboard...")

dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Otobüs Metrik Karşılaştırma - 4 Harita</title>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            text-align: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 10000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 2px;
            height: calc(100vh - 80px);
            margin-top: 80px;
            background-color: #333;
        }}
        .map-wrapper {{
            position: relative;
            background: white;
            overflow: hidden;
        }}
        .map-title {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 8px 15px;
            border-radius: 5px;
            font-weight: bold;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            font-size: 14px;
        }}
        .map-title.red {{ border-left: 4px solid #e74c3c; }}
        .map-title.blue {{ border-left: 4px solid #3498db; }}
        .map-title.green {{ border-left: 4px solid #27ae60; }}
        .map-title.orange {{ border-left: 4px solid #e67e22; }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .stats {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px 15px;
            border-radius: 5px;
            z-index: 10000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            font-size: 12px;
        }}
        .stats div {{
            margin: 3px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚌 Otobüs Metrik Karşılaştırma Dashboard</h1>
        <p>Aynı Grid Hücreleri - 4 Farklı Metrik (1km x 1km Grid - Sadece Otobüs)</p>
    </div>

    <div class="container">
        <div class="map-wrapper">
            <div class="map-title red">🔴 Yolcu Yoğunluğu</div>
            <iframe src="temp_map1_passengers.html"></iframe>
        </div>

        <div class="map-wrapper">
            <div class="map-title blue">🔵 Sefer Sayısı</div>
            <iframe src="temp_map2_trips.html"></iframe>
        </div>

        <div class="map-wrapper">
            <div class="map-title green">🟢 Araç Kapasitesi</div>
            <iframe src="temp_map3_capacity.html"></iframe>
        </div>

        <div class="map-wrapper">
            <div class="map-title orange">🟠 Doluluk Oranı</div>
            <iframe src="temp_map4_occupancy.html"></iframe>
        </div>
    </div>

    <div class="stats">
        <div><strong>Grid İstatistikleri:</strong></div>
        <div>📊 Toplam Grid: {len(df_grid):,}</div>
        <div>🚌 Otobüs Hatları: {len(df_norm_bus)}</div>
        <div>🚏 Durak Sayısı: {df_grid['stop_count'].sum():,}</div>
        <div>👥 Toplam Yolcu/Gün: {df_grid['total_passengers'].sum():,.0f}</div>
        <div>🔄 Toplam Sefer/Gün: {df_grid['total_trips'].sum():,.0f}</div>
        <div>📈 Ort. Doluluk: {df_grid['avg_occupancy'].mean():.1f}%</div>
    </div>
</body>
</html>
"""

with open('maps/bus_comparison_4maps.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print("   [SAVED] maps/bus_comparison_4maps.html")

print(f"\n{'=' * 70}")
print("4-MAP COMPARISON DASHBOARD COMPLETE")
print("=" * 70)
print("\nCreated files:")
print("  1. maps/bus_comparison_4maps.html - ANA DASHBOARD (buradan başla!)")
print("  2. maps/temp_map1_passengers.html")
print("  3. maps/temp_map2_trips.html")
print("  4. maps/temp_map3_capacity.html")
print("  5. maps/temp_map4_occupancy.html")
print("\nTarayıcıda bus_comparison_4maps.html dosyasını aç!")
print("4 haritayı aynı anda görebilirsin - aynı grid hücreleri, farklı metrikler!")
