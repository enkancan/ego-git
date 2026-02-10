"""
PDF Parser - Tüm Sayfaları Okur (Metro + Otobüs)
Gerçek tarihleri PDF içeriğinden alır.
"""

import pandas as pd
import pdfplumber
import glob
import os
import re
from datetime import datetime

# Klasör yolu - PDF'ler ana ego klasöründe
PDF_DIR = ".."
OUTPUT_FILE = "../data/ego_data_with_dates_CORRECTED.csv"

def extract_date_from_text(text):
    """Metinden DD.MM.YYYY formatında tarih çıkarır"""
    if not text:
        return None
    # Tarih Regex (DD.MM.YYYY veya D.MM.YYYY)
    date_pattern = r'(\d{1,2}\.\d{2}\.\d{4})'
    match = re.search(date_pattern, text)
    if match:
        return match.group(1)
    return None

def parse_bus_data_from_page(page, date_str):
    """Bir sayfadan otobüs verilerini parse eder"""
    data = []
    text = page.extract_text()
    if not text:
        return data
    
    # Otobüs sayfası mı kontrol et
    if 'Otobüs' not in text and 'GÜZERGÂH' not in text and 'GÜZERGAH' not in text:
        # Metro sayfası olabilir, atla
        if 'Raylı' in text or 'METRO' in text or 'TELEFERİK' in text:
            return data
    
    lines = text.split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        
        # İlk eleman hat numarası olmalı (sayı veya sayı-sayı formatı)
        hat_no = parts[0]
        if not re.match(r'^\d+(-\d+)?$', hat_no):
            continue
        
        # Son 4 sütun: Sefer, Kapasite, Yolcu, Doluluk
        try:
            doluluk = parts[-1].replace('%', '')
            if not doluluk.isdigit():
                continue
            tasinan = parts[-2].replace('.', '').replace(',', '')
            kapasite = parts[-3].replace('.', '').replace(',', '')
            sefer = parts[-4].replace('.', '').replace(',', '')
            
            # Sayısal değer kontrolü
            if not all(x.isdigit() for x in [doluluk, sefer]):
                continue
            
            # Güzergah: Hat No ile sayısal sütunlar arasındaki her şey
            guzergah = " ".join(parts[1:-4])
            
            data.append({
                'TARIH': date_str,
                'HAT NO': hat_no,
                'GÜZERGAH': guzergah,
                'SEFER SAYISI': sefer,
                'ARAÇ KAPASİTESİ': kapasite,
                'TAŞINAN YOLCU SAYISI': tasinan,
                'DOLULUK ORANI': doluluk
            })
        except:
            continue
    
    return data

def process_pdf(pdf_path):
    """Bir PDF'in TÜM sayfalarını işler"""
    all_data = []
    pdf_date = None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Önce tüm sayfalardan tarih bulmaya çalış
            for page in pdf.pages:
                text = page.extract_text()
                found_date = extract_date_from_text(text)
                if found_date:
                    pdf_date = found_date
                    break
            
            if not pdf_date:
                return [], None
            
            # Tarihi standart formata çevir
            try:
                dt = datetime.strptime(pdf_date, "%d.%m.%Y")
                formatted_date = dt.strftime("%Y-%m-%d")
            except:
                # D.MM.YYYY formatı için
                try:
                    dt = datetime.strptime(pdf_date, "%d.%m.%Y")
                    formatted_date = dt.strftime("%Y-%m-%d")
                except:
                    return [], None
            
            # Tüm sayfaları işle
            for page in pdf.pages:
                page_data = parse_bus_data_from_page(page, formatted_date)
                all_data.extend(page_data)
            
            return all_data, dt
    
    except Exception as e:
        return [], None

print("="*60)
print("PDF PROCESSOR: ALL PAGES (METRO + OTOBUS)")
print("="*60)

pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
print(f"Toplam PDF Dosyası: {len(pdf_files)}")

all_records = []
dates_found = set()
weekend_dates = set()
weekday_dates = set()

for i, f in enumerate(pdf_files):
    if i % 50 == 0:
        print(f"İşleniyor... {i}/{len(pdf_files)}")
    
    records, dt = process_pdf(f)
    
    if records and dt:
        all_records.extend(records)
        date_str = dt.strftime("%Y-%m-%d")
        dates_found.add(date_str)
        
        # Hafta sonu mu?
        if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            weekend_dates.add(date_str)
        else:
            weekday_dates.add(date_str)

print("-" * 60)
print("BİTTİ!")
print(f"Toplam Satır: {len(all_records)}")
print(f"Benzersiz Tarih: {len(dates_found)}")
print(f"Hafta İçi Tarih: {len(weekday_dates)}")
print(f"Hafta Sonu Tarih: {len(weekend_dates)}")

if weekend_dates:
    print("\nHAFTA SONU TARİHLERİ:")
    for d in sorted(weekend_dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        print(f"  {d} ({dt.strftime('%A')})")

if all_records:
    df = pd.DataFrame(all_records)
    print("\nVeri Örneği:")
    print(df.head(10))
    
    # Kaydet
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nDosya kaydedildi: {OUTPUT_FILE}")
else:
    print("Hiç veri çıkarılamadı.")
