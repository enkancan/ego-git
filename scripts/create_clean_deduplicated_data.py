"""
CLEAN DEDUPLICATED DATA CREATOR
================================
ego_data_with_dates.csv'deki duplicate kayıtları temizleyip 
ego_data_clean.csv oluşturur.

PROBLEM: 26,341 duplicate (TARIH, HAT NO) kaydı var (23.9%)
ÇÖZÜM: Aynı (TARIH, HAT NO) için SUM aggregation
"""

import pandas as pd
from datetime import datetime

print("="*80)
print("CREATING CLEAN DEDUPLICATED DATA FILE")
print("="*80)

# Load raw data
print("\n1. Loading raw data...")
df_raw = pd.read_csv('data/ego_data_with_dates.csv', encoding='utf-8-sig')

print(f"   Raw data rows: {len(df_raw):,}")
print(f"   Date range: {df_raw['TARIH'].min()} to {df_raw['TARIH'].max()}")

# Check duplicates
unique_keys = df_raw.groupby(['TARIH', 'HAT NO']).ngroups
duplicates = len(df_raw) - unique_keys

print(f"\n2. Duplicate analysis:")
print(f"   Unique (TARIH, HAT NO): {unique_keys:,}")
print(f"   Duplicate rows: {duplicates:,} ({100*duplicates/len(df_raw):.1f}%)")

# Deduplicate
print(f"\n3. Deduplicating...")
print(f"   Strategy: SUM for passengers/trips/capacity, MEAN for occupancy")

df_clean = df_raw.groupby(['TARIH', 'HAT NO'], as_index=False).agg({
    'GÜZERGAH': 'first',              # Keep first route description
    'SEFER SAYISI': 'sum',            # Sum trips (multiple time periods)
    'ARAÇ KAPASİTESİ': 'sum',         # Sum capacity
    'TAŞINAN YOLCU SAYISI': 'sum',    # Sum passengers
    'DOLULUK ORANI': 'mean'           # Average occupancy rate
})

print(f"   Clean data rows: {len(df_clean):,}")
print(f"   Removed: {len(df_raw) - len(df_clean):,} duplicate rows")

# Verify no duplicates remain
remaining_dups = df_clean.duplicated(['TARIH', 'HAT NO']).sum()
print(f"   Verification: {remaining_dups} duplicates remain (should be 0)")

# Save clean data
output_file = 'data/ego_data_clean.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n4. Saved: {output_file}")

# Create backup of raw data if not exists
import os
backup_file = 'data/ego_data_with_dates_RAW_BACKUP.csv'
if not os.path.exists(backup_file):
    df_raw.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"   Backup created: {backup_file}")
else:
    print(f"   Backup already exists: {backup_file}")

# Statistics comparison
print(f"\n" + "="*80)
print("STATISTICS COMPARISON")
print("="*80)

print(f"\n{'Metric':<30} {'Raw (Duplicated)':<20} {'Clean':<20} {'Diff':<10}")
print("-"*80)

raw_total_passengers = df_raw['TAŞINAN YOLCU SAYISI'].sum()
clean_total_passengers = df_clean['TAŞINAN YOLCU SAYISI'].sum()
print(f"{'Total passengers':<30} {raw_total_passengers:>19,} {clean_total_passengers:>19,} {raw_total_passengers - clean_total_passengers:>9,}")

raw_total_trips = df_raw['SEFER SAYISI'].sum()
clean_total_trips = df_clean['SEFER SAYISI'].sum()
print(f"{'Total trips':<30} {raw_total_trips:>19,} {clean_total_trips:>19,} {raw_total_trips - clean_total_trips:>9,}")

raw_avg_occupancy = df_raw['DOLULUK ORANI'].mean()
clean_avg_occupancy = df_clean['DOLULUK ORANI'].mean()
print(f"{'Avg occupancy':<30} {raw_avg_occupancy:>18.1f}% {clean_avg_occupancy:>18.1f}% {raw_avg_occupancy - clean_avg_occupancy:>8.1f}%")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nNow update other scripts to use: {output_file}")
print("Instead of: data/ego_data_with_dates.csv")
