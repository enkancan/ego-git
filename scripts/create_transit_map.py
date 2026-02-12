# -*- coding: utf-8 -*-
"""
Ankara Transit Network Map
Top 30 bus routes + Metro lines combined
"""
import pandas as pd
import folium
import numpy as np

print("=" * 60)
print("TRANSIT NETWORK MAP")
print("=" * 60)

# Load stop coordinates
print("Loading data...")
df_stops = pd.read_csv('data/ego_route_stops_all_coords.csv', low_memory=False)
df_stops = df_stops.dropna(subset=['final_latitude','final_longitude'])
print(f"Stops with coords: {len(df_stops)}")
print(f"Unique routes: {df_stops['hat_adi'].nunique()}")

# Get passenger data
df_bus = pd.read_csv('data/ego_data_with_dates_CORRECTED.csv', encoding='utf-8-sig')
# Find actual column names dynamically
yolcu_col = [c for c in df_bus.columns if 'YOLCU' in c][0]
doluluk_col = [c for c in df_bus.columns if 'DOLULUK' in c][0]
route_stats = df_bus.groupby('HAT NO').agg({
    yolcu_col: 'mean',
    doluluk_col: 'mean'
}).reset_index()
route_stats.columns = ['hat_no','avg_yolcu','avg_doluluk']

# Top 30
top30 = route_stats.nlargest(30, 'avg_yolcu')
print(f"Top 30 routes selected")

# === MAP ===
print("Creating map...")
m = folium.Map(location=[39.93, 32.82], zoom_start=11, tiles='cartodbdark_matter')

title_html = """<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:9999;background:rgba(0,0,0,0.85);padding:14px 28px;border-radius:10px;
    color:white;font-size:18px;font-weight:bold;font-family:Arial;">
    Ankara EGO - En Yogun 30 Otobus Hatti + Metro Agi</div>"""
m.get_root().html.add_child(folium.Element(title_html))

bus_colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD',
              '#87CEEB','#98D8C8','#F7DC6F','#BB8FCE','#85C1E9','#82E0AA',
              '#F8C471','#D7BDE2','#A3E4D7','#F9E79F','#AED6F1','#A9DFBF',
              '#FAD7A0','#D2B4DE','#A9CCE3','#ABEBC6','#F5CBA7','#D5D8DC',
              '#F1948A','#85929E','#73C6B6','#F0B27A','#C39BD3','#76D7C4']

drawn = 0
for idx, (_, row) in enumerate(top30.iterrows()):
    hat_no = str(row['hat_no'])
    route_stops = df_stops[df_stops['hat_adi'].astype(str) == hat_no].sort_values('sira')
    
    if len(route_stops) < 2:
        continue
    
    coords = list(zip(route_stops['final_latitude'], route_stops['final_longitude']))
    color = bus_colors[idx % len(bus_colors)]
    
    folium.PolyLine(
        coords, color=color, weight=3, opacity=0.7,
        tooltip=f"Hat {hat_no} - {row['avg_yolcu']:,.0f} yolcu/gun",
        popup=f"<b>Hat {hat_no}</b><br>Ort. Yolcu: {row['avg_yolcu']:,.0f}<br>Doluluk: %{row['avg_doluluk']:.0f}"
    ).add_to(m)
    drawn += 1

print(f"  Bus routes drawn: {drawn}")

# Metro lines
metro_lines = {
    'M1 Kizilay-Batikent': ('#E63946', [(39.9214,32.8537),(39.9274,32.8596),(39.9413,32.8570),(39.9549,32.8483),(39.9621,32.8437),(39.9728,32.8168),(39.9783,32.8081),(39.9840,32.7919),(39.9873,32.7779),(39.9893,32.7621),(39.9904,32.7483),(39.9716,32.7217)]),
    'M2 Kizilay-Koru': ('#D62828', [(39.9214,32.8537),(39.9197,32.8429),(39.9144,32.8277),(39.9017,32.8101),(39.8886,32.7910),(39.8807,32.7765),(39.8743,32.7495),(39.8660,32.7295),(39.8722,32.7192),(39.8819,32.6977),(39.8768,32.6849),(39.8714,32.6703)]),
    'M3 Batikent-OSB': ('#C1121F', [(39.9716,32.7217),(39.9762,32.7118),(39.9789,32.7011),(39.9803,32.6895),(39.9780,32.6775),(39.9741,32.6577),(39.9698,32.6371),(39.9661,32.6160),(39.9610,32.5995),(39.9569,32.5838),(39.9525,32.5717),(39.9478,32.5558)]),
    'M4 Kecioren': ('#457B9D', [(39.9214,32.8537),(39.9307,32.8568),(39.9370,32.8560),(39.9549,32.8483),(39.9625,32.8527),(39.9682,32.8607),(39.9726,32.8690),(39.9778,32.8695),(39.9848,32.8730),(39.9914,32.8755),(39.9986,32.8769),(40.0065,32.8803)]),
    'A1 Ankaray': ('#2A9D8F', [(39.9095,32.8139),(39.9116,32.8228),(39.9139,32.8321),(39.9153,32.8381),(39.9177,32.8470),(39.9196,32.8505),(39.9210,32.8537),(39.9214,32.8537),(39.9212,32.8650),(39.9209,32.8755),(39.9220,32.8855)]),
}

for name, (color, coords) in metro_lines.items():
    folium.PolyLine(coords, color=color, weight=7, opacity=1.0, tooltip=name).add_to(m)
    # Terminal markers
    for c in [coords[0], coords[-1]]:
        folium.CircleMarker(c, radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.9).add_to(m)

print(f"  Metro lines drawn: {len(metro_lines)}")

# Legend
legend_html = """
<div style="position:fixed;bottom:20px;right:20px;z-index:9999;
    background:rgba(20,20,20,0.92);padding:16px;border-radius:10px;
    color:white;font-family:Arial;font-size:12px;line-height:1.7;">
    <b style="font-size:14px;">Metro Hatlari</b><br>
    <span style="color:#E63946;">---</span> M1 Kizilay-Batikent<br>
    <span style="color:#D62828;">---</span> M2 Kizilay-Koru<br>
    <span style="color:#C1121F;">---</span> M3 Batikent-OSB<br>
    <span style="color:#457B9D;">---</span> M4 Kecioren<br>
    <span style="color:#2A9D8F;">---</span> A1 Ankaray<br>
    <hr style="border-color:#555;">
    <b style="font-size:14px;">En Yogun 30 Otobus Hatti</b><br>
    <small>Renkli ince cizgiler</small>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save('maps/transit_network_map.html')
print("\n[OK] maps/transit_network_map.html")

# === PNG ===
print("\nGenerating PNG...")
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import time, os
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=options)
    html_path = os.path.abspath('maps/transit_network_map.html')
    driver.get(f'file:///{html_path}')
    time.sleep(4)
    driver.save_screenshot('maps/png/transit_network_map.png')
    driver.quit()
    print("[OK] maps/png/transit_network_map.png")
except Exception as e:
    print(f"[WARN] PNG generation failed: {e}")

print("\nDone!")
