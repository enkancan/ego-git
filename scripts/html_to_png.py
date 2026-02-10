"""
Convert all HTML maps to PNG screenshots using Selenium
Includes: maps/ and ai_analysis/outputs/
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

# All HTML files to convert
jobs = [
    # maps/ -> maps/png/
    {
        'src_dir': os.path.join(BASE_DIR, 'maps'),
        'dst_dir': os.path.join(BASE_DIR, 'maps', 'png'),
        'files': [
            'grid_stop_count.html',
            'grid_yolcu_per_stop.html',
            'grid_sefer_per_stop.html',
            'grid_kapasite_per_stop.html',
            'grid_doluluk_orani.html',
            'grid_kapasite_kullanimi.html',
            'grid_stop_count_position.html',
            'grid_yolcu_per_stop_position.html',
            'grid_sefer_per_stop_position.html',
            'grid_kapasite_per_stop_position.html',
            'grid_doluluk_orani_position.html',
            'grid_kapasite_kullanimi_position.html',
            'heatmap_stop_density.html',
            'heatmap_passenger_volume.html',
            'heatmap_combined_grid.html',
            'grid_passenger_volume_squares.html',
            'grid_stop_density_squares.html',
            'grid_top100_overlay.html',
            'interactive_grid_bus_occupancy.html',
            'interactive_grid_trip_count.html',
            'interactive_grid_capacity.html',
        ]
    },
    # ai_analysis/outputs/ -> ai_analysis/outputs/png/
    {
        'src_dir': os.path.join(BASE_DIR, 'ai_analysis', 'outputs'),
        'dst_dir': os.path.join(BASE_DIR, 'ai_analysis', 'outputs', 'png'),
        'files': [
            'grid_anomaly_map.html',
            'grid_regime_shift_map.html',
            'grid_critical_map.html',
            'grid_analysis_dashboard.html',
            'top20_critical_grids.html',
        ]
    }
]

# Setup headless Chrome
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

print("="*60)
print("CONVERTING ALL HTML MAPS TO PNG")
print("="*60)

total = 0
for job in jobs:
    os.makedirs(job['dst_dir'], exist_ok=True)
    print(f"\n--- {job['src_dir']} ---")
    
    for html_file in job['files']:
        html_path = os.path.join(job['src_dir'], html_file)
        if not os.path.exists(html_path):
            print(f"  [SKIP] {html_file}")
            continue
        
        png_file = html_file.replace('.html', '.png')
        png_path = os.path.join(job['dst_dir'], png_file)
        
        abs_path = os.path.abspath(html_path)
        driver.get(f'file:///{abs_path}')
        time.sleep(3)
        
        driver.save_screenshot(png_path)
        print(f"  [OK] {png_file}")
        total += 1

driver.quit()
print(f"\nTotal {total} PNG files created.")
