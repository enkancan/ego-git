"""
Create grid-based heatmap of EGO bus stop density and passenger volume
100m x 100m grid cells over Ankara
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import json

print("=" * 70)
print("CREATING GRID-BASED HEATMAP")
print("=" * 70)

# Configuration
GRID_SIZE_METERS = 100  # 100m x 100m grid
METERS_PER_DEGREE_LAT = 111000  # Approximate
METERS_PER_DEGREE_LON = 85000   # Approximate for Ankara latitude (~40°)

GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON

# 1. Load stop coordinates
print("\n1. Loading stop coordinates...")
df_coords = pd.read_csv('data/ego_route_stops_all_coords.csv', encoding='utf-8-sig')
df_coords = df_coords[df_coords['final_latitude'].notna()].copy()

# Filter to Ankara bounds (remove outliers)
ANKARA_BOUNDS = {'lat': (39.5, 40.3), 'lon': (32.3, 33.2)}
df_coords = df_coords[
    (df_coords['final_latitude'] >= ANKARA_BOUNDS['lat'][0]) &
    (df_coords['final_latitude'] <= ANKARA_BOUNDS['lat'][1]) &
    (df_coords['final_longitude'] >= ANKARA_BOUNDS['lon'][0]) &
    (df_coords['final_longitude'] <= ANKARA_BOUNDS['lon'][1])
].copy()

print(f"   Stops with coordinates (in Ankara): {len(df_coords):,}")
print(f"   Unique stops: {df_coords['durak_kodu'].nunique():,}")

# 2. Load passenger data
print("\n2. Loading passenger data...")
df_passengers = pd.read_csv('data/ego_data_with_dates_CORRECTED.csv', encoding='utf-8-sig')

# Get average daily passengers per route
df_avg_passengers = df_passengers.groupby('HAT NO').agg({
    'TAŞINAN YOLCU SAYISI': 'mean',
    'DOLULUK ORANI': 'mean'
}).reset_index()

df_avg_passengers.columns = ['hat_adi', 'avg_daily_passengers', 'avg_occupancy']
df_avg_passengers['hat_adi'] = df_avg_passengers['hat_adi'].astype(str)

print(f"   Routes with passenger data: {len(df_avg_passengers):,}")

# 3. Merge coordinates with passenger data
print("\n3. Merging data...")
df_coords['hat_adi'] = df_coords['hat_adi'].astype(str)
df_merged = df_coords.merge(df_avg_passengers, on='hat_adi', how='left')

# Fill missing passenger data with 0
df_merged['avg_daily_passengers'] = df_merged['avg_daily_passengers'].fillna(0)
df_merged['avg_occupancy'] = df_merged['avg_occupancy'].fillna(0)

print(f"   Merged records: {len(df_merged):,}")
print(f"   With passenger data: {df_merged['avg_daily_passengers'].notna().sum():,}")

# 4. Create grid
print(f"\n4. Creating {GRID_SIZE_METERS}m x {GRID_SIZE_METERS}m grid...")

# Get bounds
lat_min = df_merged['final_latitude'].min()
lat_max = df_merged['final_latitude'].max()
lon_min = df_merged['final_longitude'].min()
lon_max = df_merged['final_longitude'].max()

print(f"   Latitude range: {lat_min:.4f} to {lat_max:.4f}")
print(f"   Longitude range: {lon_min:.4f} to {lon_max:.4f}")

# Create grid cells
def get_grid_cell(lat, lon):
    """Get grid cell coordinates for a given lat/lon"""
    cell_lat = np.floor(lat / GRID_SIZE_LAT) * GRID_SIZE_LAT
    cell_lon = np.floor(lon / GRID_SIZE_LON) * GRID_SIZE_LON
    return cell_lat, cell_lon

# Assign each stop to a grid cell
df_merged['grid_lat'] = df_merged['final_latitude'].apply(lambda lat: np.floor(lat / GRID_SIZE_LAT) * GRID_SIZE_LAT)
df_merged['grid_lon'] = df_merged['final_longitude'].apply(lambda lon: np.floor(lon / GRID_SIZE_LON) * GRID_SIZE_LON)

# 5. Aggregate by grid cell
print("\n5. Aggregating data by grid cell...")

df_grid = df_merged.groupby(['grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',  # Number of stop records in cell
    'avg_daily_passengers': 'sum',  # Total daily passengers
    'avg_occupancy': 'mean'  # Average occupancy
}).reset_index()

df_grid.columns = ['lat', 'lon', 'stop_count', 'total_passengers', 'avg_occupancy']

# Calculate cell center
df_grid['center_lat'] = df_grid['lat'] + GRID_SIZE_LAT / 2
df_grid['center_lon'] = df_grid['lon'] + GRID_SIZE_LON / 2

print(f"   Total grid cells: {len(df_grid):,}")
print(f"   Cells with stops: {len(df_grid[df_grid['stop_count'] > 0]):,}")
print(f"   Max stops per cell: {df_grid['stop_count'].max()}")
print(f"   Max passengers per cell: {df_grid['total_passengers'].max():,.0f}")

# 6. Create heatmap
print("\n6. Creating heatmap visualizations...")

# Center of Ankara
ankara_center = [39.93, 32.86]

# Heatmap 1: Stop density
print("   Creating stop density heatmap...")
m_stops = folium.Map(location=ankara_center, zoom_start=11, tiles='OpenStreetMap')

heat_data_stops = [[row['center_lat'], row['center_lon'], row['stop_count']]
                   for _, row in df_grid.iterrows()]

HeatMap(heat_data_stops,
        radius=15,
        blur=20,
        max_zoom=13).add_to(m_stops)

folium.LayerControl().add_to(m_stops)
m_stops.save('maps/heatmap_stop_density.html')
print("   [SAVED] maps/heatmap_stop_density.html")

# Heatmap 2: Passenger volume
print("   Creating passenger volume heatmap...")
m_passengers = folium.Map(location=ankara_center, zoom_start=11, tiles='OpenStreetMap')

heat_data_passengers = [[row['center_lat'], row['center_lon'], row['total_passengers']]
                        for _, row in df_grid.iterrows() if row['total_passengers'] > 0]

HeatMap(heat_data_passengers,
        radius=15,
        blur=20,
        max_zoom=13).add_to(m_passengers)

folium.LayerControl().add_to(m_passengers)
m_passengers.save('maps/heatmap_passenger_volume.html')
print("   [SAVED] maps/heatmap_passenger_volume.html")

# Heatmap 3: Combined with grid overlay
print("   Creating combined heatmap with grid overlay...")
m_combined = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

# Add heatmap
HeatMap(heat_data_passengers,
        radius=15,
        blur=20,
        max_zoom=13).add_to(m_combined)

# Add top 50 grid cells as rectangles
top_cells = df_grid.nlargest(50, 'total_passengers')

for _, cell in top_cells.iterrows():
    # Create rectangle bounds
    bounds = [[cell['lat'], cell['lon']],
              [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]]

    # Color based on passenger volume
    if cell['total_passengers'] > 50000:
        color = 'red'
    elif cell['total_passengers'] > 20000:
        color = 'orange'
    else:
        color = 'yellow'

    folium.Rectangle(
        bounds=bounds,
        color=color,
        fill=True,
        fillOpacity=0.3,
        weight=1,
        popup=f"""
        <b>Grid Cell</b><br>
        Stops: {cell['stop_count']}<br>
        Daily Passengers: {cell['total_passengers']:,.0f}<br>
        Avg Occupancy: {cell['avg_occupancy']:.1f}%
        """
    ).add_to(m_combined)

m_combined.save('maps/heatmap_combined_grid.html')
print("   [SAVED] maps/heatmap_combined_grid.html")

# Save grid data
df_grid.to_csv('data/grid_analysis.csv', index=False, encoding='utf-8-sig')
print("\n[SAVED] data/grid_analysis.csv")

# 7. Statistics
print(f"\n{'=' * 70}")
print("GRID STATISTICS")
print("=" * 70)

# Top 10 busiest cells
print("\nTop 10 busiest grid cells:")
top10 = df_grid.nlargest(10, 'total_passengers')
for idx, row in top10.iterrows():
    print(f"  {row['center_lat']:.4f}, {row['center_lon']:.4f}")
    print(f"    Stops: {row['stop_count']}, Passengers: {row['total_passengers']:,.0f}/day")

# Distribution
print(f"\nGrid cell distribution:")
print(f"  Cells with 1-5 stops: {len(df_grid[df_grid['stop_count'] <= 5]):,}")
print(f"  Cells with 6-10 stops: {len(df_grid[(df_grid['stop_count'] > 5) & (df_grid['stop_count'] <= 10)]):,}")
print(f"  Cells with 11-20 stops: {len(df_grid[(df_grid['stop_count'] > 10) & (df_grid['stop_count'] <= 20)]):,}")
print(f"  Cells with 20+ stops: {len(df_grid[df_grid['stop_count'] > 20]):,}")

print(f"\n{'=' * 70}")
print("HEATMAP CREATION COMPLETE")
print("=" * 70)
print("\nCreated maps:")
print("  1. maps/heatmap_stop_density.html - Stop density heatmap")
print("  2. maps/heatmap_passenger_volume.html - Passenger volume heatmap")
print("  3. maps/heatmap_combined_grid.html - Combined with top 50 grid cells")
