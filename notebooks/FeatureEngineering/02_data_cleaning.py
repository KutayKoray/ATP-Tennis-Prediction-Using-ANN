"""
DATA CLEANING AND PREPROCESSING
================================
This script cleans the ATP tennis match data by:
- Handling missing values intelligently
- Removing duplicates and invalid records
- Standardizing categorical variables
- Detecting and handling outliers
- Creating a clean base dataset for feature engineering

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = '../../datas/'
OUTPUT_PATH = './'
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

print("="*80)
print("ATP TENNIS MATCH DATA - CLEANING & PREPROCESSING")
print("="*80)

# --- STEP 1: LOAD ALL DATA ---
print("\n[STEP 1] Loading all historical data...")
print("-" * 80)

all_files = sorted([f for f in os.listdir(DATA_PATH) if f.startswith('atp_matches_') and f.endswith('.csv')])
print(f"Found {len(all_files)} CSV files")

dfs = []
for file in all_files:
    try:
        df = pd.read_csv(os.path.join(DATA_PATH, file))
        year = file.split('_')[-1].replace('.csv', '')
        df['data_year'] = int(year)
        dfs.append(df)
        print(f"✓ Loaded {file}: {len(df)} matches")
    except Exception as e:
        print(f"✗ Failed to load {file}: {e}")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Total dataset: {len(df_all)} matches from {df_all['data_year'].min()} to {df_all['data_year'].max()}")
print(f"   Shape: {df_all.shape}")

initial_size = len(df_all)

# --- STEP 2: REMOVE DUPLICATES ---
print("\n[STEP 2] Removing duplicate records...")
print("-" * 80)

duplicates_before = df_all.duplicated().sum()
df_all = df_all.drop_duplicates()
duplicates_removed = duplicates_before
print(f"✓ Removed {duplicates_removed} duplicate records")
print(f"   Remaining: {len(df_all)} matches")

# --- STEP 3: HANDLE CRITICAL MISSING VALUES ---
print("\n[STEP 3] Handling critical missing values...")
print("-" * 80)

# Define critical columns that must have values
critical_cols = ['winner_id', 'loser_id', 'tourney_date']

print(f"\n🔍 Checking critical columns:")
for col in critical_cols:
    missing = df_all[col].isnull().sum()
    print(f"   {col}: {missing} missing ({missing/len(df_all)*100:.2f}%)")

# Remove rows with missing critical values
before_critical = len(df_all)
df_all = df_all.dropna(subset=critical_cols)
removed_critical = before_critical - len(df_all)
print(f"\n✓ Removed {removed_critical} matches with missing critical values")
print(f"   Remaining: {len(df_all)} matches")

# --- STEP 4: CLEAN RANKING DATA ---
print("\n[STEP 4] Cleaning ranking data...")
print("-" * 80)

# Handle missing rank points - use rank to estimate if possible
def estimate_rank_points(rank):
    """Estimate rank points based on rank using approximate ATP formula"""
    if pd.isna(rank) or rank <= 0:
        return np.nan
    # Rough approximation: top players have more points
    if rank == 1:
        return 10000
    elif rank <= 10:
        return 5000 - (rank - 1) * 400
    elif rank <= 50:
        return 2000 - (rank - 10) * 30
    elif rank <= 100:
        return 800 - (rank - 50) * 10
    else:
        return max(100, 300 - (rank - 100) * 1)

# Fill missing rank points
winner_rank_missing = df_all['winner_rank_points'].isnull().sum()
loser_rank_missing = df_all['loser_rank_points'].isnull().sum()

print(f"Before cleaning:")
print(f"   Winner rank points missing: {winner_rank_missing} ({winner_rank_missing/len(df_all)*100:.2f}%)")
print(f"   Loser rank points missing: {loser_rank_missing} ({loser_rank_missing/len(df_all)*100:.2f}%)")

# Estimate missing rank points from rank
df_all['winner_rank_points'] = df_all.apply(
    lambda row: estimate_rank_points(row['winner_rank']) if pd.isna(row['winner_rank_points']) else row['winner_rank_points'],
    axis=1
)
df_all['loser_rank_points'] = df_all.apply(
    lambda row: estimate_rank_points(row['loser_rank']) if pd.isna(row['loser_rank_points']) else row['loser_rank_points'],
    axis=1
)

# If still missing, use median
winner_rank_median = df_all['winner_rank_points'].median()
loser_rank_median = df_all['loser_rank_points'].median()
df_all['winner_rank_points'].fillna(winner_rank_median, inplace=True)
df_all['loser_rank_points'].fillna(loser_rank_median, inplace=True)

print(f"\nAfter cleaning:")
print(f"   Winner rank points missing: {df_all['winner_rank_points'].isnull().sum()}")
print(f"   Loser rank points missing: {df_all['loser_rank_points'].isnull().sum()}")

# --- STEP 5: CLEAN PLAYER PHYSICAL ATTRIBUTES ---
print("\n[STEP 5] Cleaning player physical attributes...")
print("-" * 80)

# Age cleaning
print(f"\n🎂 Age statistics:")
print(f"   Winner age range: {df_all['winner_age'].min():.1f} - {df_all['winner_age'].max():.1f}")
print(f"   Loser age range: {df_all['loser_age'].min():.1f} - {df_all['loser_age'].max():.1f}")

# Remove unrealistic ages (< 14 or > 50)
df_all.loc[(df_all['winner_age'] < 14) | (df_all['winner_age'] > 50), 'winner_age'] = np.nan
df_all.loc[(df_all['loser_age'] < 14) | (df_all['loser_age'] > 50), 'loser_age'] = np.nan

# Fill missing ages with median
winner_age_median = df_all['winner_age'].median()
loser_age_median = df_all['loser_age'].median()
df_all['winner_age'].fillna(winner_age_median, inplace=True)
df_all['loser_age'].fillna(loser_age_median, inplace=True)

print(f"   Missing winner ages filled: {winner_age_median:.1f}")
print(f"   Missing loser ages filled: {loser_age_median:.1f}")

# Height cleaning
print(f"\n📏 Height statistics:")
print(f"   Winner height range: {df_all['winner_ht'].min():.0f} - {df_all['winner_ht'].max():.0f} cm")
print(f"   Loser height range: {df_all['loser_ht'].min():.0f} - {df_all['loser_ht'].max():.0f} cm")

# Remove unrealistic heights (< 160 or > 220 cm)
df_all.loc[(df_all['winner_ht'] < 160) | (df_all['winner_ht'] > 220), 'winner_ht'] = np.nan
df_all.loc[(df_all['loser_ht'] < 160) | (df_all['loser_ht'] > 220), 'loser_ht'] = np.nan

# Fill missing heights with median
winner_ht_median = df_all['winner_ht'].median()
loser_ht_median = df_all['loser_ht'].median()
df_all['winner_ht'].fillna(winner_ht_median, inplace=True)
df_all['loser_ht'].fillna(loser_ht_median, inplace=True)

print(f"   Missing winner heights filled: {winner_ht_median:.0f} cm")
print(f"   Missing loser heights filled: {loser_ht_median:.0f} cm")

# --- STEP 6: STANDARDIZE CATEGORICAL VARIABLES ---
print("\n[STEP 6] Standardizing categorical variables...")
print("-" * 80)

# Surface standardization
print(f"\n🏟️  Surface before cleaning:")
print(df_all['surface'].value_counts())

df_all['surface'] = df_all['surface'].replace('Carpet', 'Hard')  # Carpet is similar to Hard
df_all['surface'].fillna('Hard', inplace=True)  # Default to Hard if missing

print(f"\n🏟️  Surface after cleaning:")
print(df_all['surface'].value_counts())

# Hand standardization
print(f"\n✋ Hand distribution before cleaning:")
print(f"   Winner hand: {df_all['winner_hand'].value_counts().to_dict()}")
print(f"   Loser hand: {df_all['loser_hand'].value_counts().to_dict()}")

# Replace Unknown (U) with Right (R) - most common
df_all['winner_hand'] = df_all['winner_hand'].replace('U', 'R').fillna('R')
df_all['loser_hand'] = df_all['loser_hand'].replace('U', 'R').fillna('R')

print(f"\n✋ Hand distribution after cleaning:")
print(f"   Winner hand: {df_all['winner_hand'].value_counts().to_dict()}")
print(f"   Loser hand: {df_all['loser_hand'].value_counts().to_dict()}")

# --- STEP 7: CLEAN MATCH STATISTICS ---
print("\n[STEP 7] Cleaning match statistics...")
print("-" * 80)

# Service statistics
service_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']

print(f"\n🎾 Service statistics availability:")
for col in service_cols:
    if col in df_all.columns:
        available = df_all[col].notna().sum()
        pct = available / len(df_all) * 100
        print(f"   {col}: {available} ({pct:.1f}%)")

# Validate service statistics logical constraints
print(f"\n🔍 Validating service statistics logical constraints...")
if all(col in df_all.columns for col in ['w_svpt', 'w_1stIn', 'w_ace', 'w_df']):
    # Aces cannot exceed serve points
    invalid_aces_w = (df_all['w_ace'] > df_all['w_svpt']).sum()
    df_all.loc[df_all['w_ace'] > df_all['w_svpt'], 'w_ace'] = np.nan
    
    # Double faults cannot exceed serve points
    invalid_df_w = (df_all['w_df'] > df_all['w_svpt']).sum()
    df_all.loc[df_all['w_df'] > df_all['w_svpt'], 'w_df'] = np.nan
    
    # First serves in cannot exceed total serve points
    invalid_1stin_w = (df_all['w_1stIn'] > df_all['w_svpt']).sum()
    df_all.loc[df_all['w_1stIn'] > df_all['w_svpt'], 'w_1stIn'] = np.nan
    
    # First serves won cannot exceed first serves in
    invalid_1stwon_w = (df_all['w_1stWon'] > df_all['w_1stIn']).sum()
    df_all.loc[df_all['w_1stWon'] > df_all['w_1stIn'], 'w_1stWon'] = np.nan
    
    print(f"   Fixed {invalid_aces_w} invalid winner ace counts")
    print(f"   Fixed {invalid_df_w} invalid winner double fault counts")
    print(f"   Fixed {invalid_1stin_w} invalid winner 1st serve in counts")
    print(f"   Fixed {invalid_1stwon_w} invalid winner 1st serve won counts")
    
    # Same for loser
    invalid_aces_l = (df_all['l_ace'] > df_all['l_svpt']).sum()
    df_all.loc[df_all['l_ace'] > df_all['l_svpt'], 'l_ace'] = np.nan
    
    invalid_df_l = (df_all['l_df'] > df_all['l_svpt']).sum()
    df_all.loc[df_all['l_df'] > df_all['l_svpt'], 'l_df'] = np.nan
    
    invalid_1stin_l = (df_all['l_1stIn'] > df_all['l_svpt']).sum()
    df_all.loc[df_all['l_1stIn'] > df_all['l_svpt'], 'l_1stIn'] = np.nan
    
    invalid_1stwon_l = (df_all['l_1stWon'] > df_all['l_1stIn']).sum()
    df_all.loc[df_all['l_1stWon'] > df_all['l_1stIn'], 'l_1stWon'] = np.nan
    
    print(f"   Fixed {invalid_aces_l} invalid loser ace counts")
    print(f"   Fixed {invalid_df_l} invalid loser double fault counts")
    print(f"   Fixed {invalid_1stin_l} invalid loser 1st serve in counts")
    print(f"   Fixed {invalid_1stwon_l} invalid loser 1st serve won counts")

# Minutes cleaning - remove unrealistic values
if 'minutes' in df_all.columns:
    print(f"\n⏱️  Match duration:")
    print(f"   Before: {df_all['minutes'].describe()}")
    
    # Remove matches < 20 minutes or > 400 minutes (unrealistic)
    unrealistic_duration = ((df_all['minutes'] < 20) | (df_all['minutes'] > 400)).sum()
    df_all.loc[(df_all['minutes'] < 20) | (df_all['minutes'] > 400), 'minutes'] = np.nan
    
    print(f"   Fixed {unrealistic_duration} unrealistic match durations")
    print(f"   After: {df_all['minutes'].describe()}")

# --- STEP 8: HANDLE OUTLIERS ---
print("\n[STEP 8] Handling outliers...")
print("-" * 80)

def cap_outliers(series, lower_percentile=0.01, upper_percentile=0.99):
    """Cap outliers at specified percentiles"""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)

# Cap outliers in rank points (extreme values can hurt model)
print(f"\n📊 Rank points outlier handling:")
print(f"   Before - Winner: [{df_all['winner_rank_points'].min():.0f}, {df_all['winner_rank_points'].max():.0f}]")
print(f"   Before - Loser: [{df_all['loser_rank_points'].min():.0f}, {df_all['loser_rank_points'].max():.0f}]")

df_all['winner_rank_points'] = cap_outliers(df_all['winner_rank_points'])
df_all['loser_rank_points'] = cap_outliers(df_all['loser_rank_points'])

print(f"   After - Winner: [{df_all['winner_rank_points'].min():.0f}, {df_all['winner_rank_points'].max():.0f}]")
print(f"   After - Loser: [{df_all['loser_rank_points'].min():.0f}, {df_all['loser_rank_points'].max():.0f}]")

# Cap outliers in service statistics to remove data entry errors
print(f"\n🎾 Service statistics outlier handling:")
service_stat_cols = ['w_ace', 'w_df', 'l_ace', 'l_df']
for col in service_stat_cols:
    if col in df_all.columns:
        before_max = df_all[col].max()
        df_all[col] = cap_outliers(df_all[col], lower_percentile=0.0, upper_percentile=0.99)
        after_max = df_all[col].max()
        print(f"   {col}: max {before_max:.0f} → {after_max:.0f}")

# --- STEP 9: DROP UNNECESSARY COLUMNS ---
print("\n[STEP 9] Dropping unnecessary columns...")
print("-" * 80)

# Based on analysis: these columns have too much missing data or are not useful
columns_to_drop = [
    'winner_seed',   # 58% missing - not predictive
    'loser_seed',    # 75% missing - not predictive
    'winner_entry',  # 85% missing
    'loser_entry',   # 77% missing
    'score',         # Post-match info (data leakage risk)
    'best_of',       # Can be inferred from tourney_level if needed
]

dropped_cols = [col for col in columns_to_drop if col in df_all.columns]
df_all = df_all.drop(columns=dropped_cols, errors='ignore')

print(f"✓ Dropped {len(dropped_cols)} unnecessary columns:")
for col in dropped_cols:
    print(f"   - {col}")

# --- STEP 10: CREATE DATA QUALITY FLAGS ---
print("\n[STEP 10] Creating data quality flags...")
print("-" * 80)

# Flag matches with complete service statistics
service_complete_cols = ['w_ace', 'w_df', 'w_svpt', 'l_ace', 'l_df', 'l_svpt']
df_all['has_service_stats'] = df_all[service_complete_cols].notna().all(axis=1)

print(f"✓ Matches with complete service stats: {df_all['has_service_stats'].sum()} ({df_all['has_service_stats'].sum()/len(df_all)*100:.1f}%)")

# Flag matches with complete detailed service stats (for advanced features)
detailed_service_cols = service_complete_cols + ['w_1stIn', 'w_1stWon', 'w_2ndWon', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
df_all['has_detailed_service_stats'] = df_all[detailed_service_cols].notna().all(axis=1)

print(f"✓ Matches with detailed service stats: {df_all['has_detailed_service_stats'].sum()} ({df_all['has_detailed_service_stats'].sum()/len(df_all)*100:.1f}%)")

# Flag matches with rank information
df_all['has_rank_info'] = df_all[['winner_rank_points', 'loser_rank_points']].notna().all(axis=1)
print(f"✓ Matches with rank info: {df_all['has_rank_info'].sum()} ({df_all['has_rank_info'].sum()/len(df_all)*100:.1f}%)")

# --- STEP 11: FINAL DATA VALIDATION ---
print("\n[STEP 11] Final data validation...")
print("-" * 80)

# Check for any remaining critical issues
validation_checks = {
    'Negative rank points': ((df_all['winner_rank_points'] < 0) | (df_all['loser_rank_points'] < 0)).sum(),
    'Invalid ages': ((df_all['winner_age'] < 14) | (df_all['loser_age'] < 14)).sum(),
    'Invalid heights': ((df_all['winner_ht'] < 160) | (df_all['loser_ht'] < 160)).sum(),
    'Missing surface': df_all['surface'].isnull().sum(),
    'Missing hand': (df_all['winner_hand'].isnull() | df_all['loser_hand'].isnull()).sum()
}

print(f"\n🔍 Validation checks:")
all_passed = True
for check, count in validation_checks.items():
    status = "✓" if count == 0 else "✗"
    print(f"   {status} {check}: {count}")
    if count > 0:
        all_passed = False

if all_passed:
    print(f"\n✅ All validation checks passed!")
else:
    print(f"\n⚠️  Some validation checks failed - review needed")

# --- STEP 12: SAVE CLEANED DATA ---
print("\n[STEP 12] Saving cleaned dataset...")
print("-" * 80)

output_file = f'{OUTPUT_PATH}atp_matches_cleaned.csv'
df_all.to_csv(output_file, index=False)

print(f"✓ Saved cleaned dataset: {output_file}")
print(f"   Original size: {initial_size} matches")
print(f"   Cleaned size: {len(df_all)} matches")
print(f"   Removed: {initial_size - len(df_all)} matches ({(initial_size - len(df_all))/initial_size*100:.2f}%)")
print(f"   Retention rate: {len(df_all)/initial_size*100:.2f}%")

# --- STEP 13: GENERATE CLEANING SUMMARY ---
print("\n[STEP 13] Generating cleaning summary...")
print("-" * 80)

summary = f"""
ATP TENNIS MATCH DATA - CLEANING SUMMARY
========================================

Original Dataset:
- Total matches: {initial_size}
- Years: {df_all['data_year'].min()} - {df_all['data_year'].max()}

Cleaning Operations:
1. Removed {duplicates_removed} duplicate records
2. Removed {removed_critical} matches with missing critical values
3. Filled missing rank points using rank-based estimation
4. Cleaned and filled age values (median: {winner_age_median:.1f} years)
5. Cleaned and filled height values (median: {winner_ht_median:.0f} cm)
6. Standardized surface categories (merged Carpet → Hard)
7. Standardized hand categories (Unknown → Right)
8. Capped outliers in rank points
9. Validated match statistics

Final Dataset:
- Total matches: {len(df_all)}
- Retention rate: {len(df_all)/initial_size*100:.2f}%
- Data completeness: High
- Ready for feature engineering: Yes

Data Quality Flags:
- Matches with basic service stats: {df_all['has_service_stats'].sum()} ({df_all['has_service_stats'].sum()/len(df_all)*100:.1f}%)
- Matches with detailed service stats: {df_all['has_detailed_service_stats'].sum()} ({df_all['has_detailed_service_stats'].sum()/len(df_all)*100:.1f}%)
- Matches with rank information: {df_all['has_rank_info'].sum()} ({df_all['has_rank_info'].sum()/len(df_all)*100:.1f}%)

Columns Dropped:
- Removed {len(dropped_cols)} columns with high missing rates or data leakage risk

Key Improvements:
- Service statistics validated for logical consistency
- Outliers capped at 1st and 99th percentiles
- Unrealistic values removed (ages, heights, durations)
- Categorical variables standardized

Next Steps:
1. Review cleaned data: atp_matches_cleaned.csv
2. Run 03_feature_engineering.py to create advanced features
"""

with open(f'{OUTPUT_PATH}cleaning_summary.txt', 'w') as f:
    f.write(summary)

print(summary)

print("\n" + "="*80)
print("✅ DATA CLEANING COMPLETE!")
print("="*80)
print(f"\nCleaned data saved to: {output_file}")
print("Next step: Run 03_feature_engineering.py")
