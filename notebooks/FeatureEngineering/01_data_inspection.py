"""
DATA INSPECTION AND EXPLORATORY DATA ANALYSIS
==============================================
This script performs comprehensive data inspection on ATP tennis match data.
We'll analyze data quality, distributions, correlations, and identify opportunities
for feature engineering.

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- CONFIGURATION ---
DATA_PATH = '../../datas/'
OUTPUT_PATH = './'
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

print("="*80)
print("ATP TENNIS MATCH DATA - EXPLORATORY DATA ANALYSIS")
print("="*80)

# --- STEP 1: LOAD SAMPLE DATA ---
print("\n[STEP 1] Loading sample data for inspection...")
print("-" * 80)

# Load recent years for analysis (2020-2024)
sample_years = [2020, 2021, 2022, 2023, 2024]
dfs = []

for year in sample_years:
    file_path = f"{DATA_PATH}atp_matches_{year}.csv"
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
        print(f"✓ Loaded {year}: {len(df)} matches")
    except Exception as e:
        print(f"✗ Failed to load {year}: {e}")

df_sample = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Total sample size: {len(df_sample)} matches from {len(sample_years)} years")

# --- STEP 2: DATA STRUCTURE ANALYSIS ---
print("\n[STEP 2] Analyzing data structure...")
print("-" * 80)

print(f"\n📋 Dataset Shape: {df_sample.shape[0]} rows × {df_sample.shape[1]} columns")
print(f"\n📝 Column Names and Types:")
print(df_sample.dtypes)

print(f"\n🔍 First few rows:")
print(df_sample.head(3))

# --- STEP 3: MISSING VALUES ANALYSIS ---
print("\n[STEP 3] Missing values analysis...")
print("-" * 80)

missing_stats = pd.DataFrame({
    'Column': df_sample.columns,
    'Missing_Count': df_sample.isnull().sum(),
    'Missing_Percentage': (df_sample.isnull().sum() / len(df_sample) * 100).round(2),
    'Data_Type': df_sample.dtypes
})
missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print("\n⚠️  Columns with missing values:")
print(missing_stats.to_string(index=False))

# Visualize missing data
if len(missing_stats) > 0:
    plt.figure(figsize=(14, 8))
    top_missing = missing_stats.head(20)
    plt.barh(top_missing['Column'], top_missing['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Top 20 Columns with Missing Values')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}missing_values_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: missing_values_analysis.png")
    plt.close()

# --- STEP 4: CATEGORICAL FEATURES ANALYSIS ---
print("\n[STEP 4] Categorical features analysis...")
print("-" * 80)

categorical_cols = ['surface', 'tourney_level', 'round', 'winner_hand', 'loser_hand', 
                    'winner_ioc', 'loser_ioc']

for col in categorical_cols:
    if col in df_sample.columns:
        print(f"\n📊 {col.upper()} distribution:")
        value_counts = df_sample[col].value_counts()
        print(value_counts.head(10))
        print(f"   Unique values: {df_sample[col].nunique()}")
        print(f"   Missing: {df_sample[col].isnull().sum()} ({df_sample[col].isnull().sum()/len(df_sample)*100:.2f}%)")

# --- STEP 5: NUMERICAL FEATURES ANALYSIS ---
print("\n[STEP 5] Numerical features analysis...")
print("-" * 80)

numerical_cols = ['winner_age', 'loser_age', 'winner_ht', 'loser_ht', 
                  'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
                  'minutes', 'w_ace', 'w_df', 'l_ace', 'l_df']

available_numerical = [col for col in numerical_cols if col in df_sample.columns]

print("\n📈 Numerical features summary statistics:")
print(df_sample[available_numerical].describe().round(2))

# Check for outliers using IQR method
print("\n🔍 Outlier detection (IQR method):")
for col in available_numerical:
    Q1 = df_sample[col].quantile(0.25)
    Q3 = df_sample[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_sample[(df_sample[col] < lower_bound) | (df_sample[col] > upper_bound)][col]
    if len(outliers) > 0:
        print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df_sample)*100:.2f}%)")

# --- STEP 6: MATCH STATISTICS ANALYSIS ---
print("\n[STEP 6] Match statistics analysis...")
print("-" * 80)

# Service statistics
service_stats = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
available_service = [col for col in service_stats if col in df_sample.columns]

if available_service:
    print(f"\n🎾 Service statistics availability:")
    for col in available_service:
        available_pct = (1 - df_sample[col].isnull().sum() / len(df_sample)) * 100
        print(f"   {col}: {available_pct:.2f}% available")

# --- STEP 7: PLAYER RANKING ANALYSIS ---
print("\n[STEP 7] Player ranking analysis...")
print("-" * 80)

if 'winner_rank' in df_sample.columns and 'loser_rank' in df_sample.columns:
    df_sample['rank_diff'] = df_sample['winner_rank'] - df_sample['loser_rank']
    
    print(f"\n📊 Rank difference statistics:")
    print(f"   Mean: {df_sample['rank_diff'].mean():.2f}")
    print(f"   Median: {df_sample['rank_diff'].median():.2f}")
    print(f"   Std: {df_sample['rank_diff'].std():.2f}")
    
    # Analyze upsets (lower ranked player wins)
    upsets = df_sample[df_sample['rank_diff'] < 0]
    print(f"\n🎯 Upset analysis:")
    print(f"   Upsets: {len(upsets)} ({len(upsets)/len(df_sample)*100:.2f}%)")
    print(f"   Expected wins: {len(df_sample) - len(upsets)} ({(len(df_sample)-len(upsets))/len(df_sample)*100:.2f}%)")

# --- STEP 8: SURFACE ANALYSIS ---
print("\n[STEP 8] Surface-specific analysis...")
print("-" * 80)

if 'surface' in df_sample.columns:
    surface_stats = df_sample.groupby('surface').agg({
        'winner_age': 'mean',
        'minutes': 'mean',
        'w_ace': 'mean',
        'tourney_id': 'count'
    }).round(2)
    surface_stats.columns = ['Avg_Winner_Age', 'Avg_Duration_Min', 'Avg_Aces', 'Match_Count']
    print("\n🏟️  Surface-specific statistics:")
    print(surface_stats)

# --- STEP 9: TEMPORAL ANALYSIS ---
print("\n[STEP 9] Temporal patterns analysis...")
print("-" * 80)

if 'tourney_date' in df_sample.columns:
    df_sample['year'] = df_sample['tourney_date'].astype(str).str[:4].astype(int)
    df_sample['month'] = df_sample['tourney_date'].astype(str).str[4:6].astype(int)
    
    yearly_matches = df_sample.groupby('year').size()
    print(f"\n📅 Matches per year:")
    print(yearly_matches)
    
    monthly_matches = df_sample.groupby('month').size()
    print(f"\n📅 Matches per month:")
    print(monthly_matches)

# --- STEP 10: CORRELATION ANALYSIS ---
print("\n[STEP 10] Correlation analysis...")
print("-" * 80)

# Select ALL numerical columns for comprehensive correlation analysis
all_numerical_cols = [
    # Player attributes
    'winner_age', 'loser_age', 'winner_ht', 'loser_ht',
    # Rankings
    'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
    # Match duration
    'minutes',
    # Winner service stats
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    # Loser service stats
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]

# Filter to only available columns
available_corr = [col for col in all_numerical_cols if col in df_sample.columns]
print(f"\n📊 Analyzing correlations for {len(available_corr)} numerical features")

if len(available_corr) > 2:
    # Create correlation matrix (dropna to handle missing service stats)
    corr_matrix = df_sample[available_corr].corr()
    
    # Create comprehensive heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 7})
    plt.title('Comprehensive Feature Correlation Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive visualization: correlation_matrix_full.png")
    plt.close()
    
    # Also create a focused view on key features
    key_features = ['winner_age', 'loser_age', 'winner_ht', 'loser_ht', 
                    'winner_rank_points', 'loser_rank_points', 'minutes']
    available_key = [col for col in key_features if col in df_sample.columns]
    
    if len(available_key) > 2:
        corr_matrix_key = df_sample[available_key].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_key, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, annot_kws={'size': 10})
        plt.title('Key Features Correlation Matrix', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}correlation_matrix_key.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved key features visualization: correlation_matrix_key.png")
        plt.close()
    
    # Analyze strong correlations
    print("\n🔗 Strong correlations (|r| > 0.5):")
    strong_corr_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr_found = True
                print(f"   {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    if not strong_corr_found:
        print("   No strong correlations found (all |r| < 0.5)")
    
    # Analyze moderate correlations (useful for feature engineering)
    print("\n🔗 Moderate correlations (0.3 < |r| < 0.5):")
    moderate_corr_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if 0.3 < abs(corr_val) < 0.5:
                moderate_corr_found = True
                print(f"   {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    if not moderate_corr_found:
        print("   No moderate correlations in this range")
    
    # Identify potential multicollinearity issues
    print("\n⚠️  Multicollinearity check (|r| > 0.8):")
    multicollinearity_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                multicollinearity_found = True
                print(f"   WARNING: {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
                print(f"            → Consider removing one of these features")
    
    if not multicollinearity_found:
        print("   ✓ No severe multicollinearity detected")

# --- STEP 11: DATA QUALITY SUMMARY ---
print("\n[STEP 11] Data quality summary...")
print("-" * 80)

total_cells = df_sample.shape[0] * df_sample.shape[1]
missing_cells = df_sample.isnull().sum().sum()
data_completeness = (1 - missing_cells / total_cells) * 100

print(f"\n✅ Overall data quality:")
print(f"   Total cells: {total_cells:,}")
print(f"   Missing cells: {missing_cells:,}")
print(f"   Data completeness: {data_completeness:.2f}%")

# --- STEP 12: FEATURE ENGINEERING OPPORTUNITIES ---
print("\n[STEP 12] Feature engineering opportunities identified...")
print("-" * 80)

opportunities = """
🎯 RECOMMENDED FEATURE ENGINEERING STRATEGIES:

1. **Player Performance Metrics**
   - Historical win rate (overall, by surface, by tournament level)
   - Recent form (last 5, 10, 20 matches)
   - Head-to-head record between players
   - Career statistics aggregation

2. **Match Context Features**
   - Rank difference (winner_rank - loser_rank)
   - Rank points difference
   - Age difference
   - Height difference
   - Tournament importance (Grand Slam, Masters, ATP 250, etc.)

3. **Surface-Specific Features**
   - Player win rate on specific surface
   - Player's surface preference (best surface)
   - Opponent's surface weakness

4. **Service & Return Statistics** (when available)
   - First serve percentage
   - Break points conversion rate
   - Ace rate per game
   - Double fault rate

5. **Momentum & Form Features**
   - Winning/losing streak
   - Recent performance trend (improving/declining)
   - Days since last match (rest/fatigue)

6. **Interaction Features**
   - Rank difference × Surface
   - Age difference × Tournament level
   - Hand matchup (R vs L, L vs L, R vs R)

7. **Temporal Features**
   - Season phase (early/mid/late)
   - Player experience (years on tour)
   - Career stage (rising/peak/declining)

8. **Handling Missing Data**
   - Service stats: ~30-40% missing → impute with player averages
   - Rankings: Fill with median or use rank points
   - Heights/Ages: Fill with player-specific values or median
"""

print(opportunities)

# --- SAVE SUMMARY REPORT ---
print("\n[STEP 13] Saving summary report...")
print("-" * 80)

with open(f'{OUTPUT_PATH}data_inspection_report.txt', 'w') as f:
    f.write("ATP TENNIS MATCH DATA - INSPECTION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sample Size: {len(df_sample)} matches\n")
    f.write(f"Years Analyzed: {sample_years}\n")
    f.write(f"Total Features: {df_sample.shape[1]}\n")
    f.write(f"Data Completeness: {data_completeness:.2f}%\n\n")
    f.write("Missing Values Summary:\n")
    f.write(missing_stats.to_string(index=False))
    f.write("\n\n" + opportunities)

print(f"✓ Saved report: data_inspection_report.txt")

print("\n" + "="*80)
print("✅ DATA INSPECTION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review the generated visualizations and report")
print("2. Run 02_data_cleaning.py to clean the dataset")
print("3. Run 03_feature_engineering.py to create advanced features")
