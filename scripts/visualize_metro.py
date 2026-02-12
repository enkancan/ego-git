"""
Ankara Metro/Rail System Visualization
Creates interactive maps and charts from metro data
Station coordinates sourced from OpenStreetMap/Wikipedia
"""
import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

print("=" * 60)
print("METRO/RAIL SYSTEM VISUALIZATION")
print("=" * 60)

# Load data
df = pd.read_csv('data/ego_metro_data_with_dates.csv', encoding='utf-8-sig')
df['tarih'] = pd.to_datetime(df['tarih'])
df = df.sort_values('tarih')

# Clean hat_no
df['hat_no'] = df['hat_no'].replace({'T1TELEFERİK': 'T1'})

print(f"Records: {len(df)}")
print(f"Date range: {df['tarih'].min().date()} to {df['tarih'].max().date()}")
print(f"Lines: {df['hat_no'].unique().tolist()}")

os.makedirs('maps', exist_ok=True)
os.makedirs('maps/png', exist_ok=True)

# ====================================================================
# METRO STATION COORDINATES (from OpenStreetMap / Wikipedia)
# ====================================================================

metro_stations = {
    'M1': {
        'color': '#E63946',
        'name': 'M1 Kızılay – Batıkent',
        'stations': [
            ('Kızılay',       39.9214, 32.8537),
            ('Sıhhiye',       39.9274, 32.8596),
            ('Ulus',           39.9413, 32.8570),
            ('Atatürk Kültür Merkezi', 39.9549, 32.8483),
            ('Akköprü',       39.9621, 32.8437),
            ('İvedik',        39.9728, 32.8168),
            ('Yenimahalle',   39.9783, 32.8081),
            ('Demetevler',    39.9840, 32.7919),
            ('Hastane',       39.9873, 32.7779),
            ('Macunköy',      39.9893, 32.7621),
            ('OSTİM',         39.9904, 32.7483),
            ('Batıkent',      39.9716, 32.7217),
        ]
    },
    'M2': {
        'color': '#D62828',
        'name': 'M2 Kızılay – Çayyolu/Koru',
        'stations': [
            ('Kızılay',         39.9214, 32.8537),
            ('Necatibey',       39.9197, 32.8429),
            ('Milli Kütüphane', 39.9144, 32.8277),
            ('Söğütözü',        39.9017, 32.8101),
            ('MTA',             39.8886, 32.7910),
            ('ODTÜ',            39.8807, 32.7765),
            ('Bilkent',         39.8743, 32.7495),
            ('Tarım Bakanlığı', 39.8660, 32.7295),
            ('Beytepe',         39.8722, 32.7192),
            ('Ümitköy',         39.8819, 32.6977),
            ('Çayyolu',         39.8768, 32.6849),
            ('Koru',            39.8714, 32.6703),
        ]
    },
    'M3': {
        'color': '#C1121F',
        'name': 'M3 Batıkent – OSB/Törekent',
        'stations': [
            ('Batıkent',       39.9716, 32.7217),
            ('Batı Merkez',    39.9762, 32.7118),
            ('Mesa',           39.9789, 32.7011),
            ('Botanik',        39.9803, 32.6895),
            ('İstanbul Yolu',  39.9780, 32.6775),
            ('Eryaman 1-2',    39.9741, 32.6577),
            ('Eryaman 5',      39.9698, 32.6371),
            ('Devlet Mah.',    39.9661, 32.6160),
            ('Harikalar Diyarı', 39.9610, 32.5995),
            ('Fatih',          39.9569, 32.5838),
            ('GOP/Sincan',     39.9525, 32.5717),
            ('OSB-Törekent',   39.9478, 32.5558),
        ]
    },
    'M4': {
        'color': '#457B9D',
        'name': 'M4 Kızılay – Şehitler (Keçiören)',
        'stations': [
            ('Kızılay',        39.9214, 32.8537),
            ('Adliye',         39.9307, 32.8568),
            ('Gar',            39.9370, 32.8560),
            ('AKM',            39.9549, 32.8483),
            ('ASKİ',           39.9625, 32.8527),
            ('Dışkapı',        39.9682, 32.8607),
            ('Meteoroloji',    39.9726, 32.8690),
            ('Belediye',       39.9778, 32.8695),
            ('Mecidiye',       39.9848, 32.8730),
            ('Kuyubaşı',       39.9914, 32.8755),
            ('Dutluk',         39.9986, 32.8769),
            ('Şehitler',       40.0065, 32.8803),
        ]
    },
    'A1': {
        'color': '#2A9D8F',
        'name': 'A1 Ankaray (AŞTİ – Dikimevi)',
        'stations': [
            ('AŞTİ',          39.9095, 32.8139),
            ('Emek',           39.9116, 32.8228),
            ('Bahçelievler',   39.9139, 32.8321),
            ('Beşevler',       39.9153, 32.8381),
            ('Anadolu',        39.9177, 32.8470),
            ('Maltepe',        39.9196, 32.8505),
            ('Demirtepe',      39.9210, 32.8537),
            ('Kızılay',        39.9214, 32.8537),
            ('Kolej',          39.9212, 32.8650),
            ('Kurtuluş',       39.9209, 32.8755),
            ('Dikimevi',       39.9220, 32.8855),
        ]
    },
    'T1': {
        'color': '#F4A261',
        'name': 'T1 Yenimahalle–Şentepe Teleferik',
        'stations': [
            ('Yenimahalle',    39.9783, 32.8081),
            ('Şentepe',        39.9950, 32.7950),
        ]
    }
}

# Group for data: M1+M2+M3 combined in PDF data
line_group = {'M1': 'M1-M2-M3', 'M2': 'M1-M2-M3', 'M3': 'M1-M2-M3', 'M4': 'M4', 'A1': 'A1', 'T1': 'T1'}

# ============================================================
# MAP 1: Metro Lines Map
# ============================================================
print("\n📍 Creating metro lines map...")

m = folium.Map(location=[39.935, 32.79], zoom_start=12, tiles='cartodbdark_matter')

# Title
title_html = '''<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:9999;background:rgba(0,0,0,0.85);padding:14px 28px;border-radius:10px;
    color:white;font-size:20px;font-weight:bold;font-family:Arial;
    box-shadow:0 4px 12px rgba(0,0,0,0.4);">
    🚇 Ankara Raylı Sistem Ağı</div>'''
m.get_root().html.add_child(folium.Element(title_html))

# Calculate averages per line (from data)
line_stats = df.groupby('hat_no').agg({
    'tasinan_yolcu': 'mean',
    'doluluk_orani': 'mean',
    'sefer_sayisi': 'mean',
    'yolcu_kapasitesi': 'mean'
}).round(0)

for line_id, info in metro_stations.items():
    stations = info['stations']
    color = info['color']
    name = info['name']
    data_key = line_group[line_id]
    
    # Draw line
    line_coords = [(s[1], s[2]) for s in stations]
    folium.PolyLine(
        line_coords, color=color, weight=6, opacity=0.9,
        popup=f"<b>{name}</b>", tooltip=name
    ).add_to(m)
    
    # Add stations
    for i, (station_name, lat, lon) in enumerate(stations):
        is_terminal = (i == 0 or i == len(stations) - 1)
        is_transfer = station_name in ['Kızılay', 'Batıkent', 'AKM', 'Yenimahalle']
        
        # Build popup
        popup_parts = [f'<div style="font-family:Arial;min-width:220px;">']
        popup_parts.append(f'<b style="color:{color};font-size:14px;">{station_name}</b>')
        popup_parts.append(f'<br><small style="color:#888;">{name}</small>')
        
        if is_transfer:
            popup_parts.append(f'<br><span style="background:#FFD700;color:#000;padding:2px 6px;border-radius:3px;font-size:11px;">🔄 Transfer İstasyonu</span>')
        
        if data_key in line_stats.index:
            stats = line_stats.loc[data_key]
            popup_parts.append(f'<hr style="margin:6px 0;">')
            popup_parts.append(f'<b>Hat Günlük Ortalama:</b><br>')
            popup_parts.append(f'👥 Yolcu: {stats["tasinan_yolcu"]:,.0f}<br>')
            popup_parts.append(f'🚇 Sefer: {stats["sefer_sayisi"]:,.0f}<br>')
            popup_parts.append(f'📊 Doluluk: %{stats["doluluk_orani"]:.0f}<br>')
            popup_parts.append(f'💺 Kapasite: {stats["yolcu_kapasitesi"]:,.0f}')
        
        popup_parts.append('</div>')
        popup_text = ''.join(popup_parts)
        
        if is_transfer:
            folium.CircleMarker(
                [lat, lon], radius=10, color='#FFD700', fill=True,
                fill_color='#FFD700', fill_opacity=0.95, weight=2,
                popup=folium.Popup(popup_text, max_width=280),
                tooltip=f"🔄 {station_name}"
            ).add_to(m)
        elif is_terminal:
            folium.CircleMarker(
                [lat, lon], radius=8, color=color, fill=True,
                fill_color=color, fill_opacity=0.95, weight=2,
                popup=folium.Popup(popup_text, max_width=280),
                tooltip=station_name
            ).add_to(m)
        else:
            folium.CircleMarker(
                [lat, lon], radius=5, color=color, fill=True,
                fill_color='white', fill_opacity=0.9, weight=2,
                popup=folium.Popup(popup_text, max_width=280),
                tooltip=station_name
            ).add_to(m)

# Legend
legend_html = '''
<div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:rgba(30,30,30,0.95);padding:18px;border-radius:10px;
    box-shadow:0 4px 12px rgba(0,0,0,0.5);font-size:13px;color:white;
    font-family:Arial;line-height:1.7;">
    <b style="font-size:15px;">🚇 Metro Hatları</b><br>
    <span style="color:#E63946;">━━━</span> M1 Kızılay–Batıkent<br>
    <span style="color:#D62828;">━━━</span> M2 Kızılay–Koru<br>
    <span style="color:#C1121F;">━━━</span> M3 Batıkent–OSB<br>
    <span style="color:#457B9D;">━━━</span> M4 Kızılay–Şehitler<br>
    <span style="color:#2A9D8F;">━━━</span> A1 Ankaray<br>
    <span style="color:#F4A261;">━━━</span> T1 Teleferik<br>
    <hr style="border-color:#555;margin:6px 0;">
    <span style="color:#FFD700;">●</span> Transfer &nbsp;
    <span style="color:#E63946;">●</span> Terminal &nbsp;
    ○ İstasyon
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

m.save('maps/metro_lines_map.html')
print("  [OK] maps/metro_lines_map.html")

# ============================================================
# Save station coordinates as CSV
# ============================================================
print("\n💾 Saving station coordinates...")
station_rows = []
for line_id, info in metro_stations.items():
    for i, (name, lat, lon) in enumerate(info['stations']):
        station_rows.append({
            'hat': line_id,
            'hat_adi': info['name'],
            'istasyon': name,
            'sira': i + 1,
            'lat': lat,
            'lon': lon,
            'terminal': (i == 0 or i == len(info['stations']) - 1),
            'transfer': name in ['Kızılay', 'Batıkent', 'AKM', 'Yenimahalle']
        })
df_stations = pd.DataFrame(station_rows)
df_stations.to_csv('data/metro_stations.csv', index=False, encoding='utf-8-sig')
print(f"  [OK] data/metro_stations.csv ({len(df_stations)} stations)")

# ============================================================
# CHART 1: Daily Time Series
# ============================================================
print("\n📈 Creating time series charts...")

colors_data = {'M1-M2-M3': '#E63946', 'M4': '#457B9D', 'A1': '#2A9D8F', 'T1': '#F4A261'}

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=('Günlük Yolcu Sayısı', 'Doluluk Oranı (%)'),
                    vertical_spacing=0.12)

for hat in ['M1-M2-M3', 'M4', 'A1', 'T1']:
    df_hat = df[df['hat_no'] == hat].sort_values('tarih')
    if len(df_hat) == 0:
        continue
    
    df_hat['yolcu_7d'] = df_hat['tasinan_yolcu'].rolling(7, min_periods=1).mean()
    df_hat['doluluk_7d'] = df_hat['doluluk_orani'].rolling(7, min_periods=1).mean()
    
    fig.add_trace(go.Scatter(
        x=df_hat['tarih'], y=df_hat['tasinan_yolcu'],
        mode='markers', marker=dict(size=3, color=colors_data[hat], opacity=0.25),
        name=hat, legendgroup=hat, showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_hat['tarih'], y=df_hat['yolcu_7d'],
        mode='lines', line=dict(color=colors_data[hat], width=2.5),
        name=f'{hat} (7d avg)', legendgroup=hat
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_hat['tarih'], y=df_hat['doluluk_7d'],
        mode='lines', line=dict(color=colors_data[hat], width=2.5),
        name=hat, legendgroup=hat, showlegend=False
    ), row=2, col=1)

fig.update_layout(
    title='Ankara Metro/Rail — Daily Performance (315 Days)',
    height=700, template='plotly_white',
    legend=dict(orientation='h', y=1.08)
)
fig.update_yaxes(title_text='Yolcu Sayısı', row=1, col=1)
fig.update_yaxes(title_text='Doluluk %', row=2, col=1)

fig.write_html('maps/metro_daily_timeseries.html')
print("  [OK] maps/metro_daily_timeseries.html")

# ============================================================
# CHART 2: Monthly Comparison
# ============================================================
print("\n📊 Creating monthly comparison...")

df['ay'] = df['tarih'].dt.to_period('M').astype(str)
monthly = df.groupby(['ay', 'hat_no']).agg({
    'tasinan_yolcu': 'mean',
    'doluluk_orani': 'mean',
    'sefer_sayisi': 'mean'
}).reset_index()

fig2 = px.bar(monthly[monthly['hat_no'] != 'T1'], 
              x='ay', y='tasinan_yolcu', color='hat_no',
              barmode='group', color_discrete_map=colors_data,
              labels={'tasinan_yolcu': 'Avg Daily Passengers', 'ay': 'Month', 'hat_no': 'Line'},
              title='Monthly Average Daily Passengers by Line')

fig2.update_layout(height=500, template='plotly_white')
fig2.write_html('maps/metro_monthly_comparison.html')
print("  [OK] maps/metro_monthly_comparison.html")

# ============================================================
# CHART 3: Weekday vs Weekend
# ============================================================
print("\n📅 Creating weekday/weekend analysis...")

df['day_type'] = df['tarih'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

day_stats = df[df['hat_no'] != 'T1'].groupby(['hat_no', 'day_type']).agg({
    'tasinan_yolcu': 'mean',
    'doluluk_orani': 'mean',
    'sefer_sayisi': 'mean'
}).reset_index()

fig3 = make_subplots(rows=1, cols=3, 
                     subplot_titles=('Avg Passengers', 'Avg Occupancy %', 'Avg Trips'))

for i, col in enumerate(['tasinan_yolcu', 'doluluk_orani', 'sefer_sayisi'], 1):
    for hat in ['M1-M2-M3', 'M4', 'A1']:
        d = day_stats[day_stats['hat_no'] == hat]
        fig3.add_trace(go.Bar(
            x=d['day_type'], y=d[col],
            name=hat if i == 1 else None,
            marker_color=colors_data[hat],
            legendgroup=hat, showlegend=(i == 1)
        ), row=1, col=i)

fig3.update_layout(
    title='Metro: Weekday vs Weekend Performance',
    height=450, template='plotly_white',
    barmode='group', legend=dict(orientation='h', y=1.12)
)
fig3.write_html('maps/metro_weekday_weekend.html')
print("  [OK] maps/metro_weekday_weekend.html")

# ============================================================
# CHART 4: Dashboard
# ============================================================
print("\n📋 Creating metro dashboard...")

fig4 = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'pie'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'indicator'}]],
    subplot_titles=('Yolcu Payı', 'Günlük Ort. Yolcu', 'Doluluk Oranı (%)', '')
)

main_lines = df[df['hat_no'] != 'T1'].groupby('hat_no')['tasinan_yolcu'].mean()
fig4.add_trace(go.Pie(
    labels=main_lines.index, values=main_lines.values,
    marker_colors=[colors_data[h] for h in main_lines.index],
    textinfo='label+percent', hole=0.4
), row=1, col=1)

fig4.add_trace(go.Bar(
    x=main_lines.index, y=main_lines.values,
    marker_color=[colors_data[h] for h in main_lines.index],
    text=[f'{v:,.0f}' for v in main_lines.values], textposition='outside'
), row=1, col=2)

occ = df[df['hat_no'] != 'T1'].groupby('hat_no')['doluluk_orani'].mean()
fig4.add_trace(go.Bar(
    x=occ.index, y=occ.values,
    marker_color=[colors_data[h] for h in occ.index],
    text=[f'{v:.0f}%' for v in occ.values], textposition='outside'
), row=2, col=1)

total_daily = main_lines.sum()
fig4.add_trace(go.Indicator(
    mode='number',
    value=total_daily,
    title={'text': 'Toplam Günlük<br>Raylı Yolcu'},
    number={'font': {'size': 48, 'color': '#E63946'}, 'valueformat': ',.0f'},
), row=2, col=2)

fig4.update_layout(
    title='Ankara Raylı Sistem Dashboard',
    height=700, template='plotly_white', showlegend=False
)
fig4.write_html('maps/metro_dashboard.html')
print("  [OK] maps/metro_dashboard.html")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("METRO VISUALIZATION COMPLETE")
print("=" * 60)

total_stations = sum(len(info['stations']) for info in metro_stations.values())
print(f"\nStations mapped: {total_stations}")
print(f"Lines: {len(metro_stations)}")
print(f"\nGenerated files:")
print(f"  1. maps/metro_lines_map.html         - Interactive metro map ({total_stations} stations)")
print(f"  2. maps/metro_daily_timeseries.html   - Daily passengers & occupancy")
print(f"  3. maps/metro_monthly_comparison.html - Monthly comparison")
print(f"  4. maps/metro_weekday_weekend.html    - Weekday vs weekend")
print(f"  5. maps/metro_dashboard.html          - Summary dashboard")
print(f"  6. data/metro_stations.csv            - Station coordinates")
