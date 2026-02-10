"""
Ankara Metro/Rail System Visualization
Creates interactive maps and charts from metro data
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

# === Metro Station Coordinates (well-known locations) ===
metro_stations = {
    'M1-M2-M3': {
        'color': '#E63946',
        'name': 'M1-M2-M3 Çayyolu-Batıkent-Sincan Metro',
        'stations': [
            ('Çayyolu', 39.8768, 32.6849),
            ('Ümitköy', 39.8819, 32.6977),
            ('Beytepe', 39.8722, 32.7192),
            ('Tarım Bakanlığı', 39.8660, 32.7295),
            ('ODTÜ', 39.8807, 32.7365),
            ('MTA', 39.8886, 32.7510),
            ('Söğütözü', 39.9017, 32.7601),
            ('Milli Kütüphane', 39.9144, 32.7777),
            ('Necatibey', 39.9197, 32.8029),
            ('Kızılay', 39.9214, 32.8537),
            ('Sıhhiye', 39.9274, 32.8596),
            ('Ulus', 39.9413, 32.8570),
            ('Atatürk Kültür Merkezi', 39.9549, 32.8483),
            ('Akköprü', 39.9621, 32.8437),
            ('İvedik', 39.9728, 32.8168),
            ('Yenimahalle', 39.9783, 32.8081),
            ('Demetevler', 39.9840, 32.7919),
            ('Hastane', 39.9873, 32.7779),
            ('Macunköy', 39.9893, 32.7621),
            ('Ostim', 39.9904, 32.7483),
            ('Batıkent', 39.9886, 32.7238),
            ('Mesa', 39.9851, 32.7073),
            ('Botanik', 39.9827, 32.6939),
            ('İstanbul Yolu', 39.9780, 32.6775),
            ('Eryaman 1-2', 39.9741, 32.6577),
            ('Eryaman 5', 39.9698, 32.6371),
            ('Devlet Mah.', 39.9661, 32.6160),
            ('Elvankent', 39.9632, 32.5993),
            ('Fatih', 39.9569, 32.5838),
            ('Sincan Belediyesi', 39.9525, 32.5717),
        ]
    },
    'M4': {
        'color': '#457B9D',
        'name': 'M4 Keçiören Metro',
        'stations': [
            ('Atatürk Kültür Merkezi', 39.9549, 32.8483),
            ('ASKİ', 39.9625, 32.8527),
            ('Dışkapı', 39.9682, 32.8607),
            ('Meteoroloji', 39.9726, 32.8690),
            ('Belediye', 39.9815, 32.8699),
            ('Mecidiye', 39.9888, 32.8730),
            ('Kuyubaşı', 39.9954, 32.8755),
            ('Dutluk', 40.0016, 32.8769),
            ('Şehitler', 40.0085, 32.8803),
        ]
    },
    'A1': {
        'color': '#2A9D8F',
        'name': 'A1 Ankaray',
        'stations': [
            ('AŞTİ', 39.9095, 32.8139),
            ('Emek', 39.9116, 32.8228),
            ('Bahçelievler', 39.9139, 32.8321),
            ('Beşevler', 39.9153, 32.8381),
            ('Anadolu', 39.9177, 32.8470),
            ('Maltepe', 39.9205, 32.8537),
            ('Demirtepe', 39.9226, 32.8567),
            ('Kızılay', 39.9214, 32.8537),
            ('Kolej', 39.9212, 32.8650),
            ('Kurtuluş', 39.9209, 32.8755),
            ('Dikimevi', 39.9220, 32.8855),
        ]
    },
    'T1': {
        'color': '#F4A261',
        'name': 'T1 Teleferik',
        'stations': [
            ('Yenimahalle', 39.9783, 32.8081),
            ('Şentepe', 39.9950, 32.7950),
        ]
    }
}

# ============================================================
# MAP 1: Metro Lines Map with ridership info
# ============================================================
print("\n📍 Creating metro lines map...")

m = folium.Map(location=[39.935, 32.82], zoom_start=12, tiles='cartodbpositron')

# Add title
title_html = '''<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:9999;background:rgba(0,0,0,0.8);padding:12px 24px;border-radius:8px;
    color:white;font-size:18px;font-weight:bold;">
    🚇 Ankara Metro & Rail Lines</div>'''
m.get_root().html.add_child(folium.Element(title_html))

# Calculate averages per line
line_stats = df.groupby('hat_no').agg({
    'tasinan_yolcu': 'mean',
    'doluluk_orani': 'mean',
    'sefer_sayisi': 'mean',
    'yolcu_kapasitesi': 'mean'
}).round(0)

for hat_no, info in metro_stations.items():
    stations = info['stations']
    color = info['color']
    name = info['name']
    
    # Draw line
    line_coords = [(s[1], s[2]) for s in stations]
    folium.PolyLine(
        line_coords, color=color, weight=5, opacity=0.8,
        popup=f"<b>{name}</b>"
    ).add_to(m)
    
    # Add stations
    for i, (station_name, lat, lon) in enumerate(stations):
        # Get stats
        if hat_no in line_stats.index:
            stats = line_stats.loc[hat_no]
            popup_text = f"""
            <div style="font-family:Arial;min-width:200px;">
                <b style="color:{color}">{station_name}</b><br>
                <small>{name}</small><hr>
                <b>Günlük Ortalama:</b><br>
                👥 Yolcu: {stats['tasinan_yolcu']:,.0f}<br>
                🚇 Sefer: {stats['sefer_sayisi']:,.0f}<br>
                📊 Doluluk: %{stats['doluluk_orani']:.0f}<br>
                💺 Kapasite: {stats['yolcu_kapasitesi']:,.0f}
            </div>"""
        else:
            popup_text = f"<b>{station_name}</b>"
        
        # Terminal stations bigger
        if i == 0 or i == len(stations) - 1:
            folium.CircleMarker(
                [lat, lon], radius=8, color=color, fill=True,
                fill_color=color, fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=station_name
            ).add_to(m)
        else:
            folium.CircleMarker(
                [lat, lon], radius=4, color=color, fill=True,
                fill_color='white', fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=station_name
            ).add_to(m)

# Legend
legend_html = '''
<div style="position:fixed;bottom:30px;right:30px;z-index:9999;
    background:rgba(255,255,255,0.95);padding:15px;border-radius:8px;
    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:13px;">
    <b>Metro Hatları</b><br>
    <span style="color:#E63946;">━━</span> M1-M2-M3 Çayyolu-Batıkent-Sincan<br>
    <span style="color:#457B9D;">━━</span> M4 Keçiören<br>
    <span style="color:#2A9D8F;">━━</span> A1 Ankaray<br>
    <span style="color:#F4A261;">━━</span> T1 Teleferik<br>
    <br>● Terminal &nbsp; ○ İstasyon
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

m.save('maps/metro_lines_map.html')
print("  [OK] maps/metro_lines_map.html")

# ============================================================
# CHART 1: Daily Passengers Time Series
# ============================================================
print("\n📈 Creating time series charts...")

colors_map = {'M1-M2-M3': '#E63946', 'M4': '#457B9D', 'A1': '#2A9D8F', 'T1': '#F4A261'}

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=('Günlük Yolcu Sayısı', 'Doluluk Oranı (%)'),
                    vertical_spacing=0.12)

for hat in ['M1-M2-M3', 'M4', 'A1', 'T1']:
    df_hat = df[df['hat_no'] == hat].sort_values('tarih')
    if len(df_hat) == 0:
        continue
    
    # Rolling average
    df_hat['yolcu_7d'] = df_hat['tasinan_yolcu'].rolling(7, min_periods=1).mean()
    df_hat['doluluk_7d'] = df_hat['doluluk_orani'].rolling(7, min_periods=1).mean()
    
    fig.add_trace(go.Scatter(
        x=df_hat['tarih'], y=df_hat['tasinan_yolcu'],
        mode='markers', marker=dict(size=3, color=colors_map[hat], opacity=0.3),
        name=hat, legendgroup=hat, showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_hat['tarih'], y=df_hat['yolcu_7d'],
        mode='lines', line=dict(color=colors_map[hat], width=2),
        name=hat + ' (7d avg)', legendgroup=hat
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_hat['tarih'], y=df_hat['doluluk_7d'],
        mode='lines', line=dict(color=colors_map[hat], width=2),
        name=hat, legendgroup=hat, showlegend=False
    ), row=2, col=1)

fig.update_layout(
    title='Ankara Metro/Rail Daily Performance',
    height=700, template='plotly_white',
    legend=dict(orientation='h', y=1.08)
)
fig.update_yaxes(title_text='Yolcu', row=1, col=1)
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
              barmode='group',
              color_discrete_map=colors_map,
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
            marker_color=colors_map[hat],
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
# CHART 4: Overall Summary Dashboard
# ============================================================
print("\n📋 Creating metro dashboard...")

total_stats = df[df['hat_no'] != 'T1'].groupby('hat_no').agg({
    'tasinan_yolcu': ['mean', 'sum', 'std'],
    'doluluk_orani': 'mean',
    'sefer_sayisi': 'mean',
    'yolcu_kapasitesi': 'mean'
}).round(1)

fig4 = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'pie'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'indicator'}]],
    subplot_titles=('Passenger Share', 'Avg Daily Passengers', 'Occupancy Rate (%)', '')
)

# Pie chart
main_lines = df[df['hat_no'] != 'T1'].groupby('hat_no')['tasinan_yolcu'].mean()
fig4.add_trace(go.Pie(
    labels=main_lines.index, values=main_lines.values,
    marker_colors=[colors_map[h] for h in main_lines.index],
    textinfo='label+percent', hole=0.4
), row=1, col=1)

# Average passengers bar
fig4.add_trace(go.Bar(
    x=main_lines.index, y=main_lines.values,
    marker_color=[colors_map[h] for h in main_lines.index],
    text=[f'{v:,.0f}' for v in main_lines.values],
    textposition='outside'
), row=1, col=2)

# Occupancy
occ = df[df['hat_no'] != 'T1'].groupby('hat_no')['doluluk_orani'].mean()
fig4.add_trace(go.Bar(
    x=occ.index, y=occ.values,
    marker_color=[colors_map[h] for h in occ.index],
    text=[f'{v:.0f}%' for v in occ.values],
    textposition='outside'
), row=2, col=1)

# Total daily passengers indicator
total_daily = main_lines.sum()
fig4.add_trace(go.Indicator(
    mode='number+delta',
    value=total_daily,
    title={'text': 'Total Avg Daily<br>Rail Passengers'},
    number={'font': {'size': 48}, 'valueformat': ',.0f'},
), row=2, col=2)

fig4.update_layout(
    title='Ankara Metro/Rail System Dashboard',
    height=700, template='plotly_white',
    showlegend=False
)
fig4.write_html('maps/metro_dashboard.html')
print("  [OK] maps/metro_dashboard.html")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("METRO VISUALIZATION COMPLETE")
print("=" * 60)
print("\nGenerated files:")
print("  1. maps/metro_lines_map.html         - Interactive metro map")
print("  2. maps/metro_daily_timeseries.html   - Daily passengers & occupancy")
print("  3. maps/metro_monthly_comparison.html - Monthly comparison")
print("  4. maps/metro_weekday_weekend.html    - Weekday vs weekend")
print("  5. maps/metro_dashboard.html          - Summary dashboard")
