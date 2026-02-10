"""
Create square grid visualization of EGO bus data
Each 100m x 100m cell shown as colored rectangle
"""

import pandas as pd
import numpy as np
import folium
from folium import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("=" * 70)
print("CREATING SQUARE GRID MAP")
print("=" * 70)

# Configuration
GRID_SIZE_METERS = 1000  # 1km x 1km grid
METERS_PER_DEGREE_LAT = 111000
METERS_PER_DEGREE_LON = 85000

GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON

# Ankara bounds
ANKARA_BOUNDS = {'lat': (39.5, 40.3), 'lon': (32.3, 33.2)}

# 1. Load and prepare data
print("\n1. Loading data...")
df_coords = pd.read_csv('data/ego_route_stops_all_coords.csv', encoding='utf-8-sig')
df_coords = df_coords[df_coords['final_latitude'].notna()].copy()

# Filter to Ankara
df_coords = df_coords[
    (df_coords['final_latitude'] >= ANKARA_BOUNDS['lat'][0]) &
    (df_coords['final_latitude'] <= ANKARA_BOUNDS['lat'][1]) &
    (df_coords['final_longitude'] >= ANKARA_BOUNDS['lon'][0]) &
    (df_coords['final_longitude'] <= ANKARA_BOUNDS['lon'][1])
].copy()

print(f"   Stops in Ankara: {len(df_coords):,}")

# 2. Load passenger data
df_passengers = pd.read_csv('data/ego_data_with_dates_CORRECTED.csv', encoding='utf-8-sig')
df_avg_passengers = df_passengers.groupby('HAT NO').agg({
    'TAŞINAN YOLCU SAYISI': 'mean'
}).reset_index()
df_avg_passengers.columns = ['hat_adi', 'avg_daily_passengers']
df_avg_passengers['hat_adi'] = df_avg_passengers['hat_adi'].astype(str)

# 3. Merge
df_coords['hat_adi'] = df_coords['hat_adi'].astype(str)
df_merged = df_coords.merge(df_avg_passengers, on='hat_adi', how='left')
df_merged['avg_daily_passengers'] = df_merged['avg_daily_passengers'].fillna(0)

# FIX: Calculate per-stop metrics to avoid duplication
# Each route's total passengers are divided by its stop count
df_merged['hat_stop_count'] = df_merged.groupby('hat_adi')['durak_kodu'].transform('count')
df_merged['passengers_per_stop'] = df_merged['avg_daily_passengers'] / df_merged['hat_stop_count']

# 4. Create grid
print("\n2. Creating grid cells...")
df_merged['grid_lat'] = np.floor(df_merged['final_latitude'] / GRID_SIZE_LAT) * GRID_SIZE_LAT
df_merged['grid_lon'] = np.floor(df_merged['final_longitude'] / GRID_SIZE_LON) * GRID_SIZE_LON

# Aggregate by cell - use MEAN of per-stop metrics instead of SUM
df_grid = df_merged.groupby(['grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',
    'passengers_per_stop': 'mean'  # Mean of per-stop values
}).reset_index()
df_grid.columns = ['lat', 'lon', 'stop_count', 'avg_passengers_per_stop']

# Calculate total estimate for the grid (for backwards compatibility with color scale)
df_grid['total_passengers'] = df_grid['avg_passengers_per_stop'] * df_grid['stop_count']

print(f"   Grid cells: {len(df_grid):,}")
print(f"   Max stops per cell: {df_grid['stop_count'].max()}")
print(f"   Max passengers: {df_grid['total_passengers'].max():,.0f}/day")

# 5. Create color scale function
def get_color_for_passengers(passengers):
    """Get color based on passenger volume"""
    if passengers == 0:
        return '#cccccc'  # Gray for no passengers
    elif passengers < 10000:
        return '#fee5d9'  # Very light red
    elif passengers < 50000:
        return '#fcae91'  # Light red
    elif passengers < 100000:
        return '#fb6a4a'  # Medium red
    elif passengers < 250000:
        return '#de2d26'  # Dark red
    elif passengers < 500000:
        return '#a50f15'  # Very dark red
    else:
        return '#67000d'  # Darkest red

def get_color_for_stops(stop_count):
    """Get color based on stop density"""
    if stop_count == 0:
        return '#cccccc'
    elif stop_count < 5:
        return '#edf8e9'  # Very light green
    elif stop_count < 10:
        return '#bae4b3'  # Light green
    elif stop_count < 20:
        return '#74c476'  # Medium green
    elif stop_count < 50:
        return '#31a354'  # Dark green
    elif stop_count < 100:
        return '#006d2c'  # Very dark green
    else:
        return '#00441b'  # Darkest green

# 6. Create maps
ankara_center = [39.93, 32.86]

# Map 1: Passenger volume grid
print("\n3. Creating passenger volume grid map...")
m_passengers = folium.Map(
    location=ankara_center,
    zoom_start=11,
    tiles='CartoDB positron'
)

for _, cell in df_grid.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_for_passengers(cell['total_passengers'])

    folium.Rectangle(
        bounds=bounds,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=0.5,
        popup=f"""
        <b>Grid Cell</b><br>
        Stops: {cell['stop_count']}<br>
        Avg Passengers/Stop: {cell['avg_passengers_per_stop']:,.1f}<br>
        Total Est. Passengers: {cell['total_passengers']:,.0f}
        """
    ).add_to(m_passengers)

# Add legend for passengers
legend_html_passengers = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 200px; height: 250px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Günlük Yolcu Sayısı</b></p>
<p><i style="background:#67000d;width:20px;height:20px;display:inline-block;"></i> > 500,000</p>
<p><i style="background:#a50f15;width:20px;height:20px;display:inline-block;"></i> 250K - 500K</p>
<p><i style="background:#de2d26;width:20px;height:20px;display:inline-block;"></i> 100K - 250K</p>
<p><i style="background:#fb6a4a;width:20px;height:20px;display:inline-block;"></i> 50K - 100K</p>
<p><i style="background:#fcae91;width:20px;height:20px;display:inline-block;"></i> 10K - 50K</p>
<p><i style="background:#fee5d9;width:20px;height:20px;display:inline-block;"></i> < 10K</p>
<p><i style="background:#cccccc;width:20px;height:20px;display:inline-block;"></i> Veri yok</p>
</div>
'''
m_passengers.get_root().html.add_child(folium.Element(legend_html_passengers))

m_passengers.save('maps/grid_passenger_volume_squares.html')
print("   [SAVED] maps/grid_passenger_volume_squares.html")

# Map 2: Stop density grid
print("\n4. Creating stop density grid map...")
m_stops = folium.Map(
    location=ankara_center,
    zoom_start=11,
    tiles='CartoDB positron'
)

for _, cell in df_grid.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_for_stops(cell['stop_count'])

    folium.Rectangle(
        bounds=bounds,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=0.5,
        popup=f"""
        <b>Grid Cell</b><br>
        Stops: {cell['stop_count']}<br>
        Avg Passengers/Stop: {cell['avg_passengers_per_stop']:,.1f}<br>
        Total Est. Passengers: {cell['total_passengers']:,.0f}
        """
    ).add_to(m_stops)

# Add legend for stops
legend_html_stops = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 200px; height: 250px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Durak Yoğunluğu</b></p>
<p><i style="background:#00441b;width:20px;height:20px;display:inline-block;"></i> > 100 durak</p>
<p><i style="background:#006d2c;width:20px;height:20px;display:inline-block;"></i> 50 - 100</p>
<p><i style="background:#31a354;width:20px;height:20px;display:inline-block;"></i> 20 - 50</p>
<p><i style="background:#74c476;width:20px;height:20px;display:inline-block;"></i> 10 - 20</p>
<p><i style="background:#bae4b3;width:20px;height:20px;display:inline-block;"></i> 5 - 10</p>
<p><i style="background:#edf8e9;width:20px;height:20px;display:inline-block;"></i> < 5</p>
<p><i style="background:#cccccc;width:20px;height:20px;display:inline-block;"></i> Veri yok</p>
</div>
'''
m_stops.get_root().html.add_child(folium.Element(legend_html_stops))

m_stops.save('maps/grid_stop_density_squares.html')
print("   [SAVED] maps/grid_stop_density_squares.html")

# Map 3: Combined overlay (top 100 cells)
print("\n5. Creating combined overlay map...")
m_combined = folium.Map(
    location=ankara_center,
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Add top 100 passenger cells
top_cells = df_grid.nlargest(100, 'total_passengers')

for _, cell in top_cells.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_for_passengers(cell['total_passengers'])

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=0.8,
        weight=1,
        popup=f"""
        <b>Yoğun Bölge</b><br>
        Durak: {cell['stop_count']}<br>
        Günlük Yolcu: {cell['total_passengers']:,.0f}
        """
    ).add_to(m_combined)

m_combined.get_root().html.add_child(folium.Element(legend_html_passengers))
m_combined.save('maps/grid_top100_overlay.html')
print("   [SAVED] maps/grid_top100_overlay.html")

print(f"\n{'=' * 70}")
print("SQUARE GRID MAPS COMPLETE")
print("=" * 70)
print("\n3 harita oluşturuldu:")
print("  1. maps/grid_passenger_volume_squares.html - Yolcu hacmi (kırmızı tonlar)")
print("  2. maps/grid_stop_density_squares.html - Durak yoğunluğu (yeşil tonlar)")
print("  3. maps/grid_top100_overlay.html - En yoğun 100 hücre")
print(f"\nHer kare: {GRID_SIZE_METERS}m x {GRID_SIZE_METERS}m")
print(f"Toplam hücre: {len(df_grid):,}")
