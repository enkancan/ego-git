"""
Grid temelli HTML haritalar oluşturur - Feature engineering için kullanılan ham veriler
Her feature için ayrı harita oluşturur ve maps/ klasörüne kaydeder

Kullanım:
  python visualize_raw_features_grid.py
  python visualize_raw_features_grid.py --input data/daily_grid_data_position.csv --suffix _position
"""

import pandas as pd
import folium
from folium import plugins
import numpy as np
from pathlib import Path
import argparse
import sys

# Komut satırı argümanları
parser = argparse.ArgumentParser(description='Grid temelli haritalar oluştur')
parser.add_argument('--input', type=str,
                    default='data/daily_grid_data.csv',
                    help='Input CSV dosyası (default: data/daily_grid_data.csv)')
parser.add_argument('--suffix', type=str,
                    default='',
                    help='Output dosya ismine eklenecek suffix (default: yok)')

args = parser.parse_args()

# Veri yolu
DATA_PATH = Path(__file__).parent.parent / args.input
OUTPUT_DIR = Path(__file__).parent.parent / "maps"
OUTPUT_DIR.mkdir(exist_ok=True)

# Suffix
OUTPUT_SUFFIX = args.suffix

# Veriyi yükle
print(f"Veri yükleniyor: {DATA_PATH}...")
if not DATA_PATH.exists():
    print(f"HATA: Dosya bulunamadı: {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"Toplam satır sayısı: {len(df)}")
print(f"Toplam grid sayısı: {df['grid_id'].nunique()}")
print(f"Tarih aralığı: {df['tarih'].min()} - {df['tarih'].max()}")

# Her grid için ortalama değerleri hesapla (tüm günler üzerinden)
grid_avg = df.groupby(['grid_lat', 'grid_lon']).agg({
    'stop_count': 'mean',
    'yolcu_per_stop': 'mean',
    'sefer_per_stop': 'mean',
    'kapasite_per_stop': 'mean',
    'doluluk_orani': 'mean',
    'kapasite_kullanimi': 'mean'
}).reset_index()

print(f"\nGrid ortalama değerleri hesaplandı: {len(grid_avg)} grid")

# Feature tanımları (Türkçe açıklamalar)
FEATURES = {
    'stop_count': {
        'title': 'Durak Sayısı (Grid Bazlı)',
        'description': 'Her grid hücresindeki ortalama durak sayısı',
        'colormap': 'YlOrRd',
        'unit': 'durak'
    },
    'yolcu_per_stop': {
        'title': 'Durak Başına Yolcu Sayısı (Ağırlıklı)',
        'description': 'Ağırlıklı dağıtım: Hub duraklar daha fazla yolcu alır (transfer gücüne göre)',
        'colormap': 'RdYlGn',
        'unit': 'yolcu/durak'
    },
    'sefer_per_stop': {
        'title': 'Durak Başına Sefer Sayısı',
        'description': 'Her durak için ortalama günlük sefer sayısı',
        'colormap': 'Blues_Dark',  # Daha koyu tonlardan başlayan mavi
        'unit': 'sefer/durak'
    },
    'kapasite_per_stop': {
        'title': 'Durak Başına Kapasite',
        'description': 'Her durak için ortalama günlük toplam kapasite',
        'colormap': 'Purples',
        'unit': 'kapasite/durak'
    },
    'doluluk_orani': {
        'title': 'Doluluk Oranı (%)',
        'description': 'Otobüslerin ortalama doluluk oranı',
        'colormap': 'RdYlBu_r',
        'unit': '%'
    },
    'kapasite_kullanimi': {
        'title': 'Kapasite Kullanımı (%)',
        'description': 'Toplam kapasitenin ne kadarının kullanıldığı',
        'colormap': 'Spectral_r',
        'unit': '%'
    }
}

# Ankara merkez koordinatları
ANKARA_CENTER = [39.9334, 32.8597]

def create_grid_map(feature_name, feature_info):
    """Belirli bir feature için grid haritası oluştur"""

    print(f"\n{feature_info['title']} haritası oluşturuluyor...")

    # Verileri hazırla
    data = grid_avg[['grid_lat', 'grid_lon', feature_name]].copy()
    data = data.dropna()

    if len(data) == 0:
        print(f"  UYARI: {feature_name} için veri yok!")
        return

    # İstatistikler
    min_val = data[feature_name].min()
    max_val = data[feature_name].max()
    mean_val = data[feature_name].mean()
    median_val = data[feature_name].median()
    p95_val = data[feature_name].quantile(0.95)
    p99_val = data[feature_name].quantile(0.99)

    print(f"  Min: {min_val:.2f}, Max: {max_val:.2f}, Ortalama: {mean_val:.2f}, Medyan: {median_val:.2f}")
    print(f"  95th percentile: {p95_val:.2f}, 99th percentile: {p99_val:.2f}")
    print(f"  Toplam grid: {len(data)}")

    # Harita oluştur
    m = folium.Map(
        location=ANKARA_CENTER,
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Renk normalizasyonu için - Percentile-based (95th) to handle outliers
    # Outlier'lar max rengi alır, normal gridler düzgün dağılım alır
    norm_max = p95_val  # 95th percentile'ı max olarak kullan
    norm_data = (data[feature_name] - min_val) / (norm_max - min_val) if norm_max > min_val else data[feature_name] * 0
    norm_data = norm_data.clip(0, 1)  # 0-1 arasında tut

    # Grid boyutları (create_daily_grid_data.py ile aynı)
    GRID_SIZE_LAT = 1000 / 111000  # ≈0.009° (1km)
    GRID_SIZE_LON = 1000 / 85000   # ≈0.012° (1km)

    # Her grid hücresi için rectangle ekle
    for idx, row in data.iterrows():
        grid_lat = row['grid_lat']  # Sol-alt köşe (floor function sonucu)
        grid_lon = row['grid_lon']  # Sol-alt köşe (floor function sonucu)
        value = row[feature_name]
        norm_value = norm_data.iloc[idx] if isinstance(norm_data, pd.Series) else 0

        # Grid hücresinin tam sınırları (sol-alt köşeden başla, GRID_SIZE kadar ekle)
        # Bu şekilde kenarlar tam yapışır
        bounds = [
            [grid_lat, grid_lon],                                    # Sol-alt köşe
            [grid_lat + GRID_SIZE_LAT, grid_lon + GRID_SIZE_LON]   # Sağ-üst köşe
        ]

        # Renk belirleme
        color = get_color(norm_value, feature_info['colormap'])

        # Grid merkez noktası (görsel için)
        center_lat = grid_lat + GRID_SIZE_LAT / 2
        center_lon = grid_lon + GRID_SIZE_LON / 2

        # Popup bilgisi
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>{feature_info['title']}</b><br>
            <b>Değer:</b> {value:.2f} {feature_info['unit']}<br>
            <b>Grid Merkez:</b> ({center_lat:.4f}, {center_lon:.4f})<br>
            <b>Grid Boyut:</b> ~1km × 1km<br>
            <b>Percentile:</b> {norm_value*100:.1f}%
        </div>
        """

        folium.Rectangle(
            bounds=bounds,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{value:.2f} {feature_info['unit']}"
        ).add_to(m)

    # Colormap gradient'leri (legend için)
    gradient_colors = {
        'YlOrRd': 'linear-gradient(to right, #ffffcc, #ffeda0, #feb24c, #fd8d3c, #fc4e2a, #e31a1c, #b10026)',
        'RdYlGn': 'linear-gradient(to right, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850)',
        'Blues': 'linear-gradient(to right, #f7fbff, #deebf7, #c6dbef, #9ecae1, #6baed6, #4292c6, #2171b5, #084594)',
        'Blues_Dark': 'linear-gradient(to right, #9ecae1, #6baed6, #4292c6, #2171b5, #08519c, #08306b)',  # Daha koyu mavi
        'Purples': 'linear-gradient(to right, #fcfbfd, #efedf5, #dadaeb, #bcbddc, #9e9ac8, #807dba, #6a51a3, #4a1486)',
        'RdYlBu_r': 'linear-gradient(to right, #313695, #4575b4, #74add1, #fee090, #fdae61, #f46d43, #d73027)',
        'Spectral_r': 'linear-gradient(to right, #9e0142, #d53e4f, #f46d43, #fdae61, #fee08b, #e6f598, #abdda4, #66c2a5)'
    }

    gradient = gradient_colors.get(feature_info['colormap'], gradient_colors['YlOrRd'])

    # Legend ekle (HTML olarak)
    legend_html = f"""
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 240px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight: bold;">{feature_info['title']}</p>
        <p style="margin:5px 0; font-size:12px;">{feature_info['description']}</p>
        <p style="margin:5px 0; font-size:11px;">
            <b>Min:</b> {min_val:.2f} {feature_info['unit']}<br>
            <b>Max:</b> {max_val:.2f} {feature_info['unit']}<br>
            <b>95th pct:</b> {p95_val:.2f} {feature_info['unit']}<br>
            <b>Ortalama:</b> {mean_val:.2f} {feature_info['unit']}<br>
            <b>Medyan:</b> {median_val:.2f} {feature_info['unit']}<br>
            <b>Grid sayısı:</b> {len(data)}
        </p>
        <div style="background: {gradient};
                    height: 20px; border: 1px solid #999;"></div>
        <p style="margin:2px 0; font-size:10px; display:flex; justify-content:space-between;">
            <span>{min_val:.1f}</span><span>{p95_val:.1f}+</span>
        </p>
        <p style="margin:5px 0 0 0; font-size:9px; color:#666; font-style:italic;">
            * Renk skalası 95th percentile'a göre<br>
            * Outlier'lar max rengi alır
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Başlık ekle
    title_html = f"""
    <div style="position: fixed;
                top: 10px; left: 50px; width: 500px; height: 60px;
                background-color: white; z-index:9999;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <h3 style="margin:0; color: #333;">{feature_info['title']}</h3>
        <p style="margin:5px 0; font-size:12px; color: #666;">{feature_info['description']}</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Kaydet
    output_file = OUTPUT_DIR / f"grid_{feature_name}{OUTPUT_SUFFIX}.html"
    m.save(str(output_file))
    print(f"  [OK] Kaydedildi: {output_file}")

    return m

def get_color(normalized_value, colormap_name):
    """Normalize edilmiş değere göre renk döndür"""

    # Basit renk gradientleri
    colormaps = {
        'YlOrRd': ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'],
        'RdYlGn': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850'],
        'Blues': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
        'Blues_Dark': ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],  # Daha koyu mavi tonları
        'Purples': ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'],
        'RdYlBu_r': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027'],
        'Spectral_r': ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    }

    colors = colormaps.get(colormap_name, colormaps['YlOrRd'])

    # Değeri renk dizisine eşle
    idx = int(normalized_value * (len(colors) - 1))
    idx = max(0, min(idx, len(colors) - 1))

    return colors[idx]

# Tüm feature'lar için harita oluştur
def main():
    print("=" * 60)
    print("GRID TABANLI HAM FEATURE HARİTALARI OLUŞTURULUYOR")
    print("=" * 60)

    for feature_name, feature_info in FEATURES.items():
        try:
            create_grid_map(feature_name, feature_info)
        except Exception as e:
            print(f"  HATA: {feature_name} için harita oluşturulamadı: {e}")

    print("\n" + "=" * 60)
    print("TÜM HARİTALAR OLUŞTURULDU!")
    print(f"Klasör: {OUTPUT_DIR}")
    print("=" * 60)

    # Oluşturulan dosyaları listele
    print("\nOluşturulan haritalar:")
    for feature_name in FEATURES.keys():
        output_file = OUTPUT_DIR / f"grid_{feature_name}{OUTPUT_SUFFIX}.html"
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  [OK] grid_{feature_name}.html ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
