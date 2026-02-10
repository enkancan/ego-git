"""
Görselleştirme: Anomaly Detection ve Regime Shift sonuçları
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

print("=" * 80)
print("VISUALIZING ANOMALY & REGIME SHIFT RESULTS")
print("=" * 80)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load results
print("\nLoading results...")
df_combined = pd.read_csv('data/grid_combined_analysis.csv', encoding='utf-8-sig')

print(f"Total grids: {len(df_combined)}")
print(f"Anomalous grids: {df_combined['anomaly_consensus'].sum()}")
print(f"Unstable grids (regime shift): {df_combined['regime_shift_consensus'].sum()}")
print(f"Critical grids (both): {df_combined['critical_grid'].sum()}")

# ============================================================================
# VISUALIZATION 1: FOLIUM MAPS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING FOLIUM MAPS")
print("=" * 80)

GRID_SIZE_LAT = 1000 / 111000  # 1km in degrees
GRID_SIZE_LON = 1000 / 85000

ankara_center = [39.93, 32.86]

# ----------------------------------------------------------------------------
# Map 1: Anomaly Detection Results
# ----------------------------------------------------------------------------

print("\nMap 1: Anomaly Detection Results...")
m_anomaly = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

for _, row in df_combined.iterrows():
    bounds = [
        [row['grid_lat_x'], row['grid_lon_x']],
        [row['grid_lat_x'] + GRID_SIZE_LAT, row['grid_lon_x'] + GRID_SIZE_LON]
    ]

    # Color by anomaly count
    if row['anomaly_count'] == 0:
        color = '#27ae60'  # green - normal
        opacity = 0.2
    elif row['anomaly_count'] == 1:
        color = '#f39c12'  # orange - suspicious
        opacity = 0.4
    elif row['anomaly_count'] == 2:
        color = '#e67e22'  # dark orange - anomalous
        opacity = 0.6
    elif row['anomaly_count'] == 3:
        color = '#e74c3c'  # red - very anomalous
        opacity = 0.7
    else:  # 4 methods
        color = '#c0392b'  # dark red - extreme anomaly
        opacity = 0.8

    status = 'Normal' if row['anomaly_count'] == 0 else f'Anomaly ({row["anomaly_count"]} methods)'

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=opacity,
        weight=0.5,
        popup=f"""
        <b>Grid: {row['grid_id']}</b><br>
        Durum: {status}<br>
        Anomaly Methods: {row['anomaly_count']}/4<br>
        <br>
        Autoencoder: {'✓' if row['anomaly_autoencoder'] else '✗'}<br>
        Isolation Forest: {'✓' if row['anomaly_isolation_forest'] else '✗'}<br>
        LOF: {'✓' if row['anomaly_lof'] else '✗'}<br>
        Graph-based: {'✓' if row['anomaly_graph'] else '✗'}<br>
        <br>
        Avg Passengers/Stop: {row['yolcu_per_stop']:,.0f}<br>
        Avg Occupancy: {row['doluluk_orani']:.1f}%<br>
        Reconstruction Error: {row['reconstruction_error']:.3f}
        """
    ).add_to(m_anomaly)

legend_html_anomaly = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 280px; height: 280px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Hizmet-Talep Uyuşmazlığı</b></p>
<p><i style="background:#c0392b;width:30px;height:20px;display:inline-block;opacity:0.8"></i> 4 Yöntem (Ekstrem)</p>
<p><i style="background:#e74c3c;width:30px;height:20px;display:inline-block;opacity:0.7"></i> 3 Yöntem (Çok Yüksek)</p>
<p><i style="background:#e67e22;width:30px;height:20px;display:inline-block;opacity:0.6"></i> 2 Yöntem (Yüksek)</p>
<p><i style="background:#f39c12;width:30px;height:20px;display:inline-block;opacity:0.4"></i> 1 Yöntem (Şüpheli)</p>
<p><i style="background:#27ae60;width:30px;height:20px;display:inline-block;opacity:0.2"></i> Normal</p>
<p style="margin-top:10px;font-size:12px;">Autoencoder, Isolation Forest,<br>LOF, Graph-based</p>
</div>
'''
m_anomaly.get_root().html.add_child(folium.Element(legend_html_anomaly))
m_anomaly.save('ai_analysis/outputs/grid_anomaly_map.html')
print("   [SAVED] ai_analysis/outputs/grid_anomaly_map.html")

# ----------------------------------------------------------------------------
# Map 2: Regime Shift Results
# ----------------------------------------------------------------------------

print("\nMap 2: Regime Shift Results...")
m_regime = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

for _, row in df_combined.iterrows():
    bounds = [
        [row['grid_lat_x'], row['grid_lon_x']],
        [row['grid_lat_x'] + GRID_SIZE_LAT, row['grid_lon_x'] + GRID_SIZE_LON]
    ]

    # Color by regime shift status
    if row['regime_shift_embedding'] and row['regime_shift_clustering']:
        color = '#8e44ad'  # purple - both methods
        opacity = 0.8
        status = 'Rejim Kayması (Her İki Yöntem)'
    elif row['regime_shift_embedding']:
        color = '#3498db'  # blue - embedding only
        opacity = 0.6
        status = 'Rejim Kayması (Embedding)'
    elif row['regime_shift_clustering']:
        color = '#1abc9c'  # teal - clustering only
        opacity = 0.6
        status = 'Rejim Kayması (Kümeleme)'
    else:
        color = '#95a5a6'  # gray - stable
        opacity = 0.2
        status = 'Stabil'

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=opacity,
        weight=0.5,
        popup=f"""
        <b>Grid: {row['grid_id']}</b><br>
        Durum: {status}<br>
        <br>
        Embedding Instability: {row['embedding_instability']:.3f}<br>
        Cluster Switch Rate: {row['cluster_switch_rate']:.1%}<br>
        <br>
        Avg Passengers/Stop: {row['yolcu_per_stop']:,.0f}<br>
        Avg Occupancy: {row['doluluk_orani']:.1f}%
        """
    ).add_to(m_regime)

legend_html_regime = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 280px; height: 240px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Rejim Kayması (Temporal Instability)</b></p>
<p><i style="background:#8e44ad;width:30px;height:20px;display:inline-block;opacity:0.8"></i> Her İki Yöntem</p>
<p><i style="background:#3498db;width:30px;height:20px;display:inline-block;opacity:0.6"></i> Embedding Instability</p>
<p><i style="background:#1abc9c;width:30px;height:20px;display:inline-block;opacity:0.6"></i> Kümeleme Kayması</p>
<p><i style="background:#95a5a6;width:30px;height:20px;display:inline-block;opacity:0.2"></i> Stabil</p>
<p style="margin-top:10px;font-size:12px;">Günlük embedding mesafesi ve<br>küme değişim oranı</p>
</div>
'''
m_regime.get_root().html.add_child(folium.Element(legend_html_regime))
m_regime.save('ai_analysis/outputs/grid_regime_shift_map.html')
print("   [SAVED] ai_analysis/outputs/grid_regime_shift_map.html")

# ----------------------------------------------------------------------------
# Map 3: Critical Grids (Both Anomalous AND Unstable)
# ----------------------------------------------------------------------------

print("\nMap 3: Critical Grids...")
m_critical = folium.Map(location=ankara_center, zoom_start=11, tiles='CartoDB positron')

for _, row in df_combined.iterrows():
    bounds = [
        [row['grid_lat_x'], row['grid_lon_x']],
        [row['grid_lat_x'] + GRID_SIZE_LAT, row['grid_lon_x'] + GRID_SIZE_LON]
    ]

    if row['critical_grid']:
        color = '#e74c3c'  # red - critical
        opacity = 0.9
        status = 'KRİTİK (Anomali + Rejim Kayması)'
    elif row['anomaly_consensus']:
        color = '#f39c12'  # orange - anomalous only
        opacity = 0.6
        status = 'Sadece Anomali'
    elif row['regime_shift_consensus']:
        color = '#3498db'  # blue - regime shift only
        opacity = 0.6
        status = 'Sadece Rejim Kayması'
    else:
        color = '#ecf0f1'  # light gray - normal
        opacity = 0.1
        status = 'Normal'

    folium.Rectangle(
        bounds=bounds,
        color='black',
        fill=True,
        fillColor=color,
        fillOpacity=opacity,
        weight=0.5,
        popup=f"""
        <b>Grid: {row['grid_id']}</b><br>
        Durum: {status}<br>
        <br>
        Anomaly Count: {row['anomaly_count']}/4<br>
        Embedding Instability: {row['embedding_instability']:.3f}<br>
        Cluster Switch Rate: {row['cluster_switch_rate']:.1%}<br>
        <br>
        Avg Passengers/Stop: {row['yolcu_per_stop']:,.0f}<br>
        Avg Trips/Stop: {row['sefer_per_stop']:.0f}<br>
        Avg Occupancy: {row['doluluk_orani']:.1f}%
        """
    ).add_to(m_critical)

legend_html_critical = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 280px; height: 220px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:10px;"><b>Kritik Grid Analizi</b></p>
<p><i style="background:#e74c3c;width:30px;height:20px;display:inline-block;opacity:0.9"></i> KRİTİK (Anomali + Rejim)</p>
<p><i style="background:#f39c12;width:30px;height:20px;display:inline-block;opacity:0.6"></i> Sadece Anomali</p>
<p><i style="background:#3498db;width:30px;height:20px;display:inline-block;opacity:0.6"></i> Sadece Rejim Kayması</p>
<p><i style="background:#ecf0f1;width:30px;height:20px;display:inline-block;opacity:0.1"></i> Normal</p>
</div>
'''
m_critical.get_root().html.add_child(folium.Element(legend_html_critical))
m_critical.save('ai_analysis/outputs/grid_critical_map.html')
print("   [SAVED] ai_analysis/outputs/grid_critical_map.html")

# ============================================================================
# VISUALIZATION 2: PLOTLY CHARTS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING PLOTLY CHARTS")
print("=" * 80)

# ----------------------------------------------------------------------------
# Chart 1: Anomaly Method Comparison
# ----------------------------------------------------------------------------

print("\nChart 1: Anomaly Method Comparison...")

method_counts = {
    'Autoencoder': df_combined['anomaly_autoencoder'].sum(),
    'Isolation Forest': df_combined['anomaly_isolation_forest'].sum(),
    'LOF': df_combined['anomaly_lof'].sum(),
    'Graph-based': df_combined['anomaly_graph'].sum()
}

fig1 = go.Figure(data=[
    go.Bar(
        x=list(method_counts.keys()),
        y=list(method_counts.values()),
        marker_color=['#e74c3c', '#3498db', '#f39c12', '#9b59b6'],
        text=list(method_counts.values()),
        textposition='auto'
    )
])

fig1.update_layout(
    title='Anomaly Detection Methods Comparison',
    xaxis_title='Method',
    yaxis_title='Anomalies Detected',
    height=500
)

fig1.write_html('ai_analysis/outputs/anomaly_methods_comparison.html')
print("   [SAVED] ai_analysis/outputs/anomaly_methods_comparison.html")

# ----------------------------------------------------------------------------
# Chart 2: Anomaly Score Distributions
# ----------------------------------------------------------------------------

print("\nChart 2: Anomaly Score Distributions...")

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Reconstruction Error', 'Isolation Forest Score',
                    'LOF Score', 'Graph Anomaly Score')
)

fig2.add_trace(
    go.Histogram(x=df_combined['reconstruction_error'], name='Recon Error',
                 marker_color='#e74c3c', nbinsx=30),
    row=1, col=1
)

fig2.add_trace(
    go.Histogram(x=df_combined['anomaly_score_if'], name='IF Score',
                 marker_color='#3498db', nbinsx=30),
    row=1, col=2
)

fig2.add_trace(
    go.Histogram(x=df_combined['lof_score'], name='LOF Score',
                 marker_color='#f39c12', nbinsx=30),
    row=2, col=1
)

fig2.add_trace(
    go.Histogram(x=df_combined['graph_anomaly_score'], name='Graph Score',
                 marker_color='#9b59b6', nbinsx=30),
    row=2, col=2
)

fig2.update_layout(height=700, title_text="Anomaly Score Distributions", showlegend=False)
fig2.write_html('ai_analysis/outputs/anomaly_score_distributions.html')
print("   [SAVED] ai_analysis/outputs/anomaly_score_distributions.html")

# ----------------------------------------------------------------------------
# Chart 3: Regime Shift Analysis
# ----------------------------------------------------------------------------

print("\nChart 3: Regime Shift Analysis...")

regime_categories = {
    'Stabil': ((~df_combined['regime_shift_embedding']) & (~df_combined['regime_shift_clustering'])).sum(),
    'Sadece Embedding': (df_combined['regime_shift_embedding'] & (~df_combined['regime_shift_clustering'])).sum(),
    'Sadece Kümeleme': ((~df_combined['regime_shift_embedding']) & df_combined['regime_shift_clustering']).sum(),
    'Her İkisi': (df_combined['regime_shift_embedding'] & df_combined['regime_shift_clustering']).sum()
}

fig3 = go.Figure(data=[
    go.Pie(
        labels=list(regime_categories.keys()),
        values=list(regime_categories.values()),
        marker=dict(colors=['#95a5a6', '#3498db', '#1abc9c', '#8e44ad']),
        hole=0.3
    )
])

fig3.update_layout(
    title='Rejim Kayması Dağılımı',
    height=500
)

fig3.write_html('ai_analysis/outputs/regime_shift_distribution.html')
print("   [SAVED] ai_analysis/outputs/regime_shift_distribution.html")

# ----------------------------------------------------------------------------
# Chart 4: Scatter - Instability vs Anomaly
# ----------------------------------------------------------------------------

print("\nChart 4: Instability vs Anomaly Scatter...")

df_combined['category'] = 'Normal'
df_combined.loc[df_combined['anomaly_consensus'] & (~df_combined['regime_shift_consensus']), 'category'] = 'Sadece Anomali'
df_combined.loc[(~df_combined['anomaly_consensus']) & df_combined['regime_shift_consensus'], 'category'] = 'Sadece Rejim'
df_combined.loc[df_combined['critical_grid'], 'category'] = 'KRİTİK'

fig4 = px.scatter(
    df_combined,
    x='embedding_instability',
    y='anomaly_count',
    color='category',
    color_discrete_map={
        'Normal': '#95a5a6',
        'Sadece Anomali': '#f39c12',
        'Sadece Rejim': '#3498db',
        'KRİTİK': '#e74c3c'
    },
    size='yolcu_per_stop',
    hover_data=['grid_id', 'doluluk_orani', 'cluster_switch_rate'],
    title='Embedding Instability vs Anomaly Count',
    labels={
        'embedding_instability': 'Embedding Instability',
        'anomaly_count': 'Anomaly Methods Count',
        'yolcu_per_stop': 'Avg Passengers/Stop'
    }
)

fig4.update_layout(height=700)
fig4.write_html('ai_analysis/outputs/instability_vs_anomaly_scatter.html')
print("   [SAVED] ai_analysis/outputs/instability_vs_anomaly_scatter.html")

# ----------------------------------------------------------------------------
# Chart 5: Top 20 Critical Grids
# ----------------------------------------------------------------------------

print("\nChart 5: Top 20 Critical Grids...")

df_critical = df_combined[df_combined['critical_grid']].copy()
df_critical['criticality_score'] = (
    df_critical['anomaly_count'] +
    2 * df_critical['embedding_instability'] +
    2 * df_critical['cluster_switch_rate']
)

top20 = df_critical.nlargest(20, 'criticality_score')

fig5 = go.Figure()

fig5.add_trace(go.Bar(
    x=top20['grid_id'],
    y=top20['anomaly_count'],
    name='Anomaly Count',
    marker_color='#e74c3c'
))

fig5.add_trace(go.Bar(
    x=top20['grid_id'],
    y=top20['embedding_instability'] / 5,  # scale down for visibility
    name='Instability (scaled /5)',
    marker_color='#3498db'
))

fig5.update_layout(
    title='Top 20 Critical Grids (Anomaly + Instability)',
    xaxis_title='Grid ID',
    yaxis_title='Score',
    barmode='group',
    height=600,
    xaxis={'tickangle': -45}
)

fig5.write_html('ai_analysis/outputs/top20_critical_grids.html')
print("   [SAVED] ai_analysis/outputs/top20_critical_grids.html")

# ============================================================================
# DASHBOARD HTML
# ============================================================================

print("\n" + "=" * 80)
print("CREATING DASHBOARD")
print("=" * 80)

total_grids = len(df_combined)
anomaly_grids = df_combined['anomaly_consensus'].sum()
regime_grids = df_combined['regime_shift_consensus'].sum()
critical_grids = df_combined['critical_grid'].sum()

dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Grid Anomaly & Regime Shift Analysis</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; padding: 30px; text-align: center; border-radius: 10px; }}
        .header h1 {{ margin: 0; font-size: 32px; }}
        .header p {{ margin: 5px 0 0 0; font-size: 16px; opacity: 0.9; }}
        .stats {{ display: flex; justify-content: space-around; margin: 30px 0; flex-wrap: wrap; }}
        .stat-box {{ background: white; padding: 25px; border-radius: 10px;
                     box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
                     min-width: 180px; margin: 10px; }}
        .stat-number {{ font-size: 42px; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; }}
        .red {{ color: #e74c3c; }}
        .orange {{ color: #f39c12; }}
        .blue {{ color: #3498db; }}
        .purple {{ color: #8e44ad; }}
        .section {{ background: white; padding: 25px; margin: 20px 0; border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #2c3e50; margin-top: 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .links a {{ display: block; padding: 12px 20px; margin: 8px 0; background: #3498db;
                    color: white; text-decoration: none; border-radius: 5px; transition: 0.3s; }}
        .links a:hover {{ background: #2980b9; transform: translateX(5px); }}
        .info {{ background: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 8px;
                 border-left: 4px solid #3498db; }}
        .warning {{ background: #fff3cd; padding: 20px; margin: 20px 0; border-radius: 8px;
                    border-left: 4px solid #f39c12; }}
        .critical {{ background: #f8d7da; padding: 20px; margin: 20px 0; border-radius: 8px;
                     border-left: 4px solid #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Grid Anomaly & Regime Shift Analysis</h1>
        <p>Hizmet-Talep Uyuşmazlığı ve Temporal İstikrarsızlık Analizi</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{total_grids}</div>
            <div class="stat-label">Toplam Grid</div>
        </div>
        <div class="stat-box">
            <div class="stat-number orange">{anomaly_grids}</div>
            <div class="stat-label">Anomali (Uyuşmazlık)</div>
        </div>
        <div class="stat-box">
            <div class="stat-number blue">{regime_grids}</div>
            <div class="stat-label">Rejim Kayması</div>
        </div>
        <div class="stat-box">
            <div class="stat-number red">{critical_grids}</div>
            <div class="stat-label">KRİTİK (Her İkisi)</div>
        </div>
    </div>

    <div class="section">
        <h2>🗺️ İnteraktif Haritalar</h2>
        <div class="links">
            <a href="grid_anomaly_map.html" target="_blank">
                🔴 Anomaly Detection Map (Hizmet-Talep Uyuşmazlığı)
            </a>
            <a href="grid_regime_shift_map.html" target="_blank">
                🔵 Regime Shift Map (Temporal İstikrarsızlık)
            </a>
            <a href="grid_critical_map.html" target="_blank">
                ⚠️ Critical Grids Map (Anomali + Rejim Kayması)
            </a>
        </div>
    </div>

    <div class="section">
        <h2>📊 İnteraktif Analizler</h2>
        <div class="links">
            <a href="anomaly_methods_comparison.html" target="_blank">
                📊 Anomaly Method Comparison
            </a>
            <a href="anomaly_score_distributions.html" target="_blank">
                📈 Anomaly Score Distributions
            </a>
            <a href="regime_shift_distribution.html" target="_blank">
                🥧 Regime Shift Distribution
            </a>
            <a href="instability_vs_anomaly_scatter.html" target="_blank">
                🔍 Instability vs Anomaly Scatter
            </a>
            <a href="top20_critical_grids.html" target="_blank">
                🏆 Top 20 Critical Grids
            </a>
        </div>
    </div>

    <div class="critical">
        <h3>⚠️ KRİTİK BULGULAR</h3>
        <p><strong>{critical_grids} grid</strong> hem hizmet-talep uyuşmazlığı hem de temporal istikrarsızlık gösteriyor.</p>
        <p>Bu gridler:</p>
        <ul>
            <li>Sistemin "normal" kabul ettiği talep-hizmet ilişkisine uymuyorlar</li>
            <li>Günler boyunca tutarsız davranış sergiliyorlar (rejim kayması)</li>
            <li>Öncelikli müdahale gerektiriyorlar</li>
        </ul>
    </div>

    <div class="info">
        <h3>ℹ️ METODOLOJI</h3>
        <h4>Bölüm 1: Hizmet-Talep Uyuşmazlığı (Unsupervised Anomaly Detection)</h4>
        <p><strong>Autoencoder (PCA):</strong> Sistemin "normal" talep-hizmet ilişkisini öğrenir. Yüksek reconstruction error = uyumsuzluk</p>
        <p><strong>Isolation Forest:</strong> Çok boyutlu uzayda izole gridleri tespit eder</p>
        <p><strong>Local Outlier Factor (LOF):</strong> Benzer gridlere göre anormal olanları bulur</p>
        <p><strong>Graph-based:</strong> Komşu gridlerle karşılaştırarak mekânsal uyumsuzlukları ortaya çıkarır</p>

        <h4>Bölüm 2: Rejim Kayması (Temporal Stability Analysis)</h4>
        <p><strong>Embedding Stability:</strong> Her gridin günlük embedding'leri arasındaki mesafeyi ölçer. Yüksek mesafe = istikrarsızlık</p>
        <p><strong>Daily Clustering:</strong> Her gün gridleri kümeler, küme değişim oranını hesaplar. Sık değişim = rejim kayması</p>
    </div>

    <div class="warning">
        <h3>💡 YORUMLAMA</h3>
        <p><strong>Anomali:</strong> Grid'in talep-hizmet dengesi sistemin normu dışında</p>
        <p><strong>Rejim Kayması:</strong> Grid aynı mekân olmasına rağmen, sistemin temsil uzayında stabil değil</p>
        <p><strong>Kritik Grid:</strong> Hem uyumsuz hem istikrarsız - makine semiyotiği açısından eşik problemi</p>
        <p style="margin-top:15px;"><em>"Sistem aynı mekânı her gün farklı bir kategoriye itiyor"</em></p>
    </div>

    <div class="section">
        <h2>📁 Veri Dosyaları</h2>
        <ul>
            <li><code>data/grid_anomaly_results.csv</code> - Anomaly detection sonuçları</li>
            <li><code>data/grid_regime_shift_results.csv</code> - Regime shift sonuçları</li>
            <li><code>data/grid_combined_analysis.csv</code> - Birleştirilmiş analiz</li>
        </ul>
    </div>
</body>
</html>
"""

with open('ai_analysis/outputs/grid_analysis_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print("\n[SAVED] ai_analysis/outputs/grid_analysis_dashboard.html")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  Maps:")
print("    1. ai_analysis/outputs/grid_anomaly_map.html")
print("    2. ai_analysis/outputs/grid_regime_shift_map.html")
print("    3. ai_analysis/outputs/grid_critical_map.html")
print("\n  Charts:")
print("    4. ai_analysis/outputs/anomaly_methods_comparison.html")
print("    5. ai_analysis/outputs/anomaly_score_distributions.html")
print("    6. ai_analysis/outputs/regime_shift_distribution.html")
print("    7. ai_analysis/outputs/instability_vs_anomaly_scatter.html")
print("    8. ai_analysis/outputs/top20_critical_grids.html")
print("\n  Dashboard:")
print("    9. ai_analysis/outputs/grid_analysis_dashboard.html")
print("\nTarayıcıda ai_analysis/outputs/grid_analysis_dashboard.html dosyasını aç!")
