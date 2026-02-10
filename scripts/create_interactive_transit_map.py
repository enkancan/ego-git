"""
Create interactive HTML map showing normalized metro + bus density
"""

import pandas as pd
import numpy as np
import folium
from folium import Rectangle
import plotly.express as px
import plotly.graph_objects as go

print("=" * 70)
print("CREATING INTERACTIVE TRANSIT MAP")
print("=" * 70)

# 1. Load normalized data
print("\n1. Loading normalized transit data...")
df_norm = pd.read_csv('data/normalized_all_transit.csv', encoding='utf-8-sig')
print(f"   Total lines: {len(df_norm)}")

# 2. Load stop coordinates (for spatial mapping)
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
    df_norm_exp[['hat_no', 'avg_yolcu', 'normalized_yolcu', 'percentile', 'category', 'mod']],
    left_on='hat_base',
    right_on='hat_no',
    how='left'
)

# Fill missing
df_merged['avg_yolcu'] = df_merged['avg_yolcu'].fillna(0)
df_merged['normalized_yolcu'] = df_merged['normalized_yolcu'].fillna(0)
df_merged['mod'] = df_merged['mod'].fillna('Otobus')
df_merged['category'] = df_merged['category'].fillna('Veri yok')

print(f"   Merged records: {len(df_merged):,}")
print(f"   Metro stops: {len(df_merged[df_merged['mod']=='Metro']):,}")
print(f"   Bus stops: {len(df_merged[df_merged['mod']=='Otobus']):,}")

# 4. Create 1km grid with normalized data
print("\n3. Creating 1km grid with normalized passenger data...")

GRID_SIZE_METERS = 1000
METERS_PER_DEGREE_LAT = 111000
METERS_PER_DEGREE_LON = 85000

GRID_SIZE_LAT = GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
GRID_SIZE_LON = GRID_SIZE_METERS / METERS_PER_DEGREE_LON

df_merged['grid_lat'] = np.floor(df_merged['final_latitude'] / GRID_SIZE_LAT) * GRID_SIZE_LAT
df_merged['grid_lon'] = np.floor(df_merged['final_longitude'] / GRID_SIZE_LON) * GRID_SIZE_LON

# Aggregate by grid cell
df_grid = df_merged.groupby(['grid_lat', 'grid_lon']).agg({
    'durak_kodu': 'count',
    'avg_yolcu': 'sum',  # Sum of all passengers in cell
    'normalized_yolcu': 'mean',  # Average normalized score
    'mod': lambda x: 'Metro+Bus' if len(set(x)) > 1 else x.iloc[0]
}).reset_index()

df_grid.columns = ['lat', 'lon', 'stop_count', 'total_passengers', 'avg_normalized', 'primary_mode']

# Normalize the grid total passengers
grid_mean = df_grid['total_passengers'].mean()
grid_std = df_grid['total_passengers'].std()
df_grid['grid_normalized'] = (df_grid['total_passengers'] - grid_mean) / grid_std

print(f"   Grid cells: {len(df_grid):,}")
print(f"   Max passengers per cell: {df_grid['total_passengers'].max():,.0f}")

# 5. Create interactive HTML map
print("\n4. Creating interactive map...")

ankara_center = [39.93, 32.86]
m = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

# Color function based on normalized passenger volume
def get_color_normalized(normalized_val):
    """Get color based on normalized passenger volume (z-score)"""
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

# Add grid cells
for _, cell in df_grid.iterrows():
    bounds = [
        [cell['lat'], cell['lon']],
        [cell['lat'] + GRID_SIZE_LAT, cell['lon'] + GRID_SIZE_LON]
    ]

    color = get_color_normalized(cell['grid_normalized'])

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=0.5,
        popup=f"""
        <b>1km Grid Hücresi</b><br>
        Durak: {cell['stop_count']}<br>
        Toplam Yolcu/Gün: {cell['total_passengers']:,.0f}<br>
        Normalize Skor: {cell['grid_normalized']:.2f}<br>
        Mod: {cell['primary_mode']}
        """
    ).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 250px; height: 280px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Normalize Edilmiş Yoğunluk</b></p>
<p><i style="background:#67000d;width:30px;height:20px;display:inline-block;"></i> Çok Yüksek (>2σ)</p>
<p><i style="background:#a50f15;width:30px;height:20px;display:inline-block;"></i> Yüksek (1-2σ)</p>
<p><i style="background:#de2d26;width:30px;height:20px;display:inline-block;"></i> Ortanın Üstü (0.5-1σ)</p>
<p><i style="background:#fb6a4a;width:30px;height:20px;display:inline-block;"></i> Orta (-0.5-0.5σ)</p>
<p><i style="background:#fcae91;width:30px;height:20px;display:inline-block;"></i> Ortanın Altı (-0.5-0σ)</p>
<p><i style="background:#fee5d9;width:30px;height:20px;display:inline-block;"></i> Düşük (<-0.5σ)</p>
<p style="margin-top:10px;"><small>σ = Standart Sapma<br>0 = Ortalama Yoğunluk</small></p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

m.save('maps/interactive_transit_density.html')
print("   [SAVED] maps/interactive_transit_density.html")

# 6. Create Plotly interactive charts
print("\n5. Creating Plotly interactive charts...")

# Chart 1: Top 50 lines interactive bar chart
top50 = df_norm.nlargest(50, 'avg_yolcu')

fig = px.bar(top50,
             x='avg_yolcu',
             y='hat_no',
             color='mod',
             color_discrete_map={'Metro': 'red', 'Otobüs': 'steelblue'},
             orientation='h',
             title='En Yoğun 50 Hat (Metro + Otobüs)',
             labels={'avg_yolcu': 'Ortalama Günlük Yolcu', 'hat_no': 'Hat No', 'mod': 'Mod'},
             hover_data=['avg_doluluk', 'avg_sefer', 'percentile'])

fig.update_layout(height=1200, showlegend=True)
fig.write_html('maps/interactive_top50_lines.html')
print("   [SAVED] maps/interactive_top50_lines.html")

# Chart 2: Scatter plot - Passengers vs Occupancy
fig2 = px.scatter(df_norm,
                  x='avg_yolcu',
                  y='avg_doluluk',
                  color='mod',
                  size='avg_sefer',
                  color_discrete_map={'Metro': 'red', 'Otobüs': 'steelblue'},
                  hover_data=['hat_no', 'category', 'percentile'],
                  title='Yolcu Sayısı vs Doluluk Oranı (Tüm Hatlar)',
                  labels={'avg_yolcu': 'Ortalama Günlük Yolcu',
                          'avg_doluluk': 'Doluluk Oranı (%)',
                          'avg_sefer': 'Sefer Sayısı',
                          'mod': 'Mod'})

fig2.add_hline(y=70, line_dash="dash", line_color="red",
               annotation_text="Yoğun Eşik (70%)")
fig2.update_layout(height=700)
fig2.write_html('maps/interactive_scatter_transit.html')
print("   [SAVED] maps/interactive_scatter_transit.html")

# Chart 3: Distribution histogram
fig3 = go.Figure()

fig3.add_trace(go.Histogram(
    x=df_norm[df_norm['mod'] == 'Otobüs']['avg_yolcu'],
    name='Otobüs',
    opacity=0.7,
    marker_color='steelblue',
    nbinsx=50
))

fig3.add_trace(go.Histogram(
    x=df_norm[df_norm['mod'] == 'Metro']['avg_yolcu'],
    name='Metro',
    opacity=0.7,
    marker_color='red',
    nbinsx=10
))

fig3.update_layout(
    title='Yolcu Dağılımı (Metro vs Otobüs)',
    xaxis_title='Ortalama Günlük Yolcu',
    yaxis_title='Hat Sayısı',
    barmode='overlay',
    height=600
)

fig3.write_html('maps/interactive_distribution.html')
print("   [SAVED] maps/interactive_distribution.html")

# 7. Create summary HTML dashboard
print("\n6. Creating summary dashboard...")

dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EGO Ulaşım Analizi - Normalize Edilmiş Veriler</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 200px; }}
        .stat-number {{ font-size: 36px; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        .links {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .links h2 {{ color: #2c3e50; }}
        .links a {{ display: block; padding: 10px; margin: 5px 0; background: #3498db; color: white; text-decoration: none; border-radius: 4px; }}
        .links a:hover {{ background: #2980b9; }}
        .metro {{ color: #e74c3c; }}
        .bus {{ color: #3498db; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚇🚌 EGO Ulaşım Sistemi - Normalize Edilmiş Analiz</h1>
        <p>Metro + Otobüs Yoğunluk Analizi (Tüm Veriler Ortalamasına Göre Normalize Edilmiş)</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">642</div>
            <div class="stat-label">Toplam Hat</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">41.6M</div>
            <div class="stat-label">Günlük Yolcu</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">64,862</div>
            <div class="stat-label">Ortalama Yolcu/Hat</div>
        </div>
        <div class="stat-box metro">
            <div class="stat-number">5</div>
            <div class="stat-label">Metro Hattı</div>
        </div>
        <div class="stat-box bus">
            <div class="stat-number">637</div>
            <div class="stat-label">Otobüs Hattı</div>
        </div>
    </div>

    <div class="links">
        <h2>📊 İnteraktif Görselleştirmeler</h2>
        <a href="interactive_transit_density.html" target="_blank">
            🗺️ İnteraktif Harita - 1km Grid Yoğunluk Haritası
        </a>
        <a href="interactive_top50_lines.html" target="_blank">
            📊 En Yoğun 50 Hat - İnteraktif Bar Chart
        </a>
        <a href="interactive_scatter_transit.html" target="_blank">
            📈 Yolcu vs Doluluk - İnteraktif Scatter Plot
        </a>
        <a href="interactive_distribution.html" target="_blank">
            📉 Yolcu Dağılımı - Metro vs Otobüs
        </a>
    </div>

    <div class="links">
        <h2>🏆 En Yoğun 10 Hat</h2>
        <p><strong>1.</strong> Hat 765 (Otobüs) - 632,358 yolcu/gün</p>
        <p><strong>2.</strong> Hat 645 (Otobüs) - 620,344 yolcu/gün</p>
        <p><strong>3.</strong> Hat 767 (Otobüs) - 619,372 yolcu/gün</p>
        <p><strong>4.</strong> Hat 782 (Otobüs) - 618,796 yolcu/gün</p>
        <p><strong>5.</strong> Hat 789 (Otobüs) - 605,034 yolcu/gün</p>
        <p><strong>6.</strong> Hat 768 (Otobüs) - 590,875 yolcu/gün</p>
        <p><strong>7.</strong> Hat 783 (Otobüs) - 587,121 yolcu/gün</p>
        <p><strong>8.</strong> Hat 790 (Otobüs) - 586,642 yolcu/gün</p>
        <p><strong>9.</strong> Hat 764 (Otobüs) - 583,111 yolcu/gün</p>
        <p><strong>10.</strong> Hat 774 (Otobüs) - 582,628 yolcu/gün</p>
    </div>

    <div class="links">
        <h2>ℹ️ Normalizasyon Bilgisi</h2>
        <p>Tüm hatlar (metro + otobüs) genel ortalamaya göre normalize edilmiştir.</p>
        <p><strong>Z-Score:</strong> Her hattın genel ortalamadan kaç standart sapma uzakta olduğunu gösterir.</p>
        <p><strong>0:</strong> Ortalama yoğunluk</p>
        <p><strong>+1:</strong> Ortalamadan 1 standart sapma yukarı</p>
        <p><strong>-1:</strong> Ortalamadan 1 standart sapma aşağı</p>
    </div>
</body>
</html>
"""

with open('maps/transit_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print("   [SAVED] maps/transit_dashboard.html")

print(f"\n{'=' * 70}")
print("INTERACTIVE MAPS COMPLETE")
print("=" * 70)
print("\nCreated files:")
print("  1. maps/transit_dashboard.html - Ana dashboard (buradan başla!)")
print("  2. maps/interactive_transit_density.html - 1km grid harita")
print("  3. maps/interactive_top50_lines.html - Top 50 hatlar")
print("  4. maps/interactive_scatter_transit.html - Yolcu vs doluluk")
print("  5. maps/interactive_distribution.html - Dağılım histogramı")
print("\nTarayıcıda transit_dashboard.html dosyasını aç!")
