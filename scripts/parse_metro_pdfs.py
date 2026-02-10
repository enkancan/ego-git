"""
Parse metro/rail system data from EGO PDFs
Extracts: M1-M2-M3, M4, A1 Ankaray, T1 Teleferik
"""

import PyPDF2
import pandas as pd
import os
import re
from datetime import datetime

print("=" * 70)
print("PARSING METRO/RAIL SYSTEM DATA FROM PDFs")
print("=" * 70)

# PDF ID to date mapping (same as bus data)
pdf_id_to_date = {}

def parse_metro_from_pdf(pdf_path):
    """Extract metro/rail system data from a PDF"""
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)

            # Get all text from PDF
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text()

            # Find the rail systems section
            # Look for lines that start with M1, M4, A1, T1
            lines = full_text.split('\n')

            metro_data = []

            for line in lines:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check if line starts with metro/rail codes
                if line.startswith(('M1-', 'M4', 'A1', 'T1')):
                    # Try to parse the line
                    parts = line.split()

                    if len(parts) >= 6:
                        try:
                            # Extract line code
                            hat_no = parts[0]

                            # Find where numbers start
                            # Numbers are typically: sefer_sayisi, tren_sayisi, kapasite, yolcu, doluluk%
                            numbers = []
                            hat_name = []

                            for i, part in enumerate(parts[1:], 1):
                                # Remove commas and % signs
                                clean_part = part.replace('.', '').replace(',', '').replace('%', '')

                                # Try to convert to number
                                try:
                                    num = int(clean_part)
                                    numbers.append(num)
                                except ValueError:
                                    # This is part of the name
                                    if len(numbers) == 0:  # Still reading the name
                                        hat_name.append(part)

                            # We expect 5 numbers: sefer, tren, kapasite, yolcu, doluluk
                            if len(numbers) >= 5:
                                metro_data.append({
                                    'hat_no': hat_no,
                                    'hat_adi': ' '.join(hat_name) if hat_name else '',
                                    'sefer_sayisi': numbers[0],
                                    'tren_sayisi': numbers[1],
                                    'yolcu_kapasitesi': numbers[2],
                                    'tasinan_yolcu': numbers[3],
                                    'doluluk_orani': numbers[4]
                                })
                        except Exception as e:
                            # Skip problematic lines
                            continue

            return metro_data

    except Exception as e:
        print(f"[ERROR] {pdf_path}: {str(e)[:100]}")
        return []

# Get all PDFs
pdf_files = sorted([f for f in os.listdir('pdfs') if f.endswith('.pdf')])
print(f"\nFound {len(pdf_files)} PDF files")

# Parse all PDFs
all_metro_data = []
success_count = 0
fail_count = 0

print("\nParsing PDFs...")
for idx, pdf_file in enumerate(pdf_files, 1):
    pdf_path = os.path.join('pdfs', pdf_file)
    pdf_id = pdf_file.replace('.pdf', '')

    metro_data = parse_metro_from_pdf(pdf_path)

    if metro_data:
        # Add date and PDF ID to each record
        for record in metro_data:
            record['pdf_id'] = pdf_id
            all_metro_data.append(record)

        success_count += 1

        if idx % 50 == 0:
            print(f"  [{idx}/{len(pdf_files)}] Processed... ({success_count} OK, {fail_count} FAIL)")
    else:
        fail_count += 1

print(f"\n{'=' * 70}")
print("PARSING COMPLETE")
print("=" * 70)
print(f"PDFs processed: {len(pdf_files)}")
print(f"Successful: {success_count}")
print(f"Failed: {fail_count}")
print(f"Total metro records: {len(all_metro_data):,}")

# Convert to DataFrame
df = pd.DataFrame(all_metro_data)

if len(df) > 0:
    # Add date mapping
    print(f"\nAdding date information...")

    # Try to extract date from PDF text (similar to bus parsing)
    # For now, use PDF ID as identifier
    df['tarih'] = None

    # Show summary
    print(f"\n{'=' * 70}")
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(df):,}")
    print(f"Unique lines: {df['hat_no'].nunique()}")
    print(f"\nLines found:")
    for line in df['hat_no'].unique():
        count = len(df[df['hat_no'] == line])
        avg_passengers = df[df['hat_no'] == line]['tasinan_yolcu'].mean()
        avg_occupancy = df[df['hat_no'] == line]['doluluk_orani'].mean()
        print(f"  {line}: {count} records, avg {avg_passengers:,.0f} passengers/day, {avg_occupancy:.1f}% occupancy")

    # Save to CSV
    df.to_csv('data/ego_metro_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n[SAVED] data/ego_metro_data.csv")

    # Show sample data
    print(f"\nSample data:")
    print(df.head(10).to_string())

else:
    print("\n[WARNING] No metro data extracted!")

print(f"\n{'=' * 70}")
