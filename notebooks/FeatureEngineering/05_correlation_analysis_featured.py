"""
CORRELATION ANALYSIS - FEATURE-ENGINEERED DATASET
==================================================
This script creates correlation matrices for the feature-engineered dataset
to visualize relationships between features for presentation purposes.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
INPUT_FILE = './atp_matches_featured.csv'
OUTPUT_PATH = './'
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

print("="*80)
print("CORRELATION ANALYSIS - FEATURE-ENGINEERED DATASET")
print("="*80)

# --- LOAD DATA ---
print("\n[STEP 1] Loading feature-engineered data...")
print("-" * 80)

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df)} matches")
print(f"✓ Total columns: {len(df.columns)}")

# --- SELECT NUMERICAL FEATURES ---
print("\n[STEP 2] Selecting numerical features for correlation analysis...")
print("-" * 80)

# All engineered features (51 features)
numerical_features = [
    # Basic player attributes
    'winner_rank_points', 'loser_rank_points',
    'winner_age', 'loser_age',
    'winner_ht', 'loser_ht',
    'winner_experience', 'loser_experience',
    
    # Historical performance
    'winner_career_win_rate', 'loser_career_win_rate',
    'winner_surface_win_rate', 'loser_surface_win_rate',
    'winner_recent_form', 'loser_recent_form',
    
    # Historical service statistics
    'winner_avg_ace_rate', 'loser_avg_ace_rate',
    'winner_avg_df_rate', 'loser_avg_df_rate',
    'winner_avg_1st_serve_pct', 'loser_avg_1st_serve_pct',
    'winner_avg_1st_serve_win_pct', 'loser_avg_1st_serve_win_pct',
    
    # Head-to-head
    'h2h_matches', 'h2h_win_rate',
    
    # Tournament context
    'tourney_importance',
    'surface_hard', 'surface_clay', 'surface_grass',
    
    # Hand matchup
    'both_right_handed', 'both_left_handed', 'mixed_handed',
    'winner_is_lefty', 'loser_is_lefty',
    
    # Temporal
    'month', 'quarter', 'month_sin', 'month_cos',
]

# Filter only available features
available_features = [f for f in numerical_features if f in df.columns]
print(f"✓ Selected {len(available_features)} numerical features")

# --- COMPUTE CORRELATION MATRIX ---
print("\n[STEP 3] Computing correlation matrix...")
print("-" * 80)

df_corr = df[available_features].corr()
print(f"✓ Correlation matrix shape: {df_corr.shape}")

# --- VISUALIZATION 1: FULL CORRELATION MATRIX ---
print("\n[STEP 4] Creating full correlation matrix visualization...")
print("-" * 80)

plt.figure(figsize=(20, 18))
mask = np.triu(np.ones_like(df_corr, dtype=bool))  # Mask upper triangle

sns.heatmap(df_corr, 
            mask=mask,
            annot=False,  # Too many features for annotations
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})

plt.title('Correlation Matrix - Feature-Engineered Dataset (51 Features)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

output_file = OUTPUT_PATH + 'correlation_matrix_featured_full.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# --- VISUALIZATION 2: KEY FEATURES CORRELATION ---
print("\n[STEP 5] Creating key features correlation matrix...")
print("-" * 80)

# Select most important features for clearer visualization
key_features = [
    'winner_rank_points', 'loser_rank_points',
    'winner_age', 'loser_age',
    'winner_career_win_rate', 'loser_career_win_rate',
    'winner_surface_win_rate', 'loser_surface_win_rate',
    'winner_recent_form', 'loser_recent_form',
    'winner_avg_ace_rate', 'loser_avg_ace_rate',
    'winner_avg_1st_serve_pct', 'loser_avg_1st_serve_pct',
    'h2h_win_rate',
    'tourney_importance',
    'surface_hard', 'surface_clay',
]

# Filter available key features
available_key_features = [f for f in key_features if f in df.columns]
df_corr_key = df[available_key_features].corr()

plt.figure(figsize=(14, 12))
mask_key = np.triu(np.ones_like(df_corr_key, dtype=bool))

sns.heatmap(df_corr_key, 
            mask=mask_key,
            annot=True,  # Show values for key features
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot_kws={'fontsize': 8})

plt.title('Correlation Matrix - Key Features (Top 18)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

output_file_key = OUTPUT_PATH + 'correlation_matrix_featured_key.png'
plt.savefig(output_file_key, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file_key}")
plt.close()

# --- VISUALIZATION 3: SERVICE STATISTICS CORRELATION ---
print("\n[STEP 6] Creating service statistics correlation matrix...")
print("-" * 80)

service_features = [
    'winner_avg_ace_rate', 'loser_avg_ace_rate',
    'winner_avg_df_rate', 'loser_avg_df_rate',
    'winner_avg_1st_serve_pct', 'loser_avg_1st_serve_pct',
    'winner_avg_1st_serve_win_pct', 'loser_avg_1st_serve_win_pct',
    'winner_career_win_rate', 'loser_career_win_rate',
    'winner_rank_points', 'loser_rank_points',
]

available_service_features = [f for f in service_features if f in df.columns]
df_corr_service = df[available_service_features].corr()

plt.figure(figsize=(12, 10))
mask_service = np.triu(np.ones_like(df_corr_service, dtype=bool))

sns.heatmap(df_corr_service, 
            mask=mask_service,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            annot_kws={'fontsize': 9})

plt.title('Correlation Matrix - Service Statistics & Performance', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

output_file_service = OUTPUT_PATH + 'correlation_matrix_service_stats.png'
plt.savefig(output_file_service, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file_service}")
plt.close()

# --- ANALYSIS: STRONG CORRELATIONS ---
print("\n[STEP 7] Analyzing strong correlations...")
print("-" * 80)

# Find strong correlations (>0.7 or <-0.7)
strong_corr = []
for i in range(len(df_corr.columns)):
    for j in range(i+1, len(df_corr.columns)):
        corr_value = df_corr.iloc[i, j]
        if abs(corr_value) > 0.7:
            strong_corr.append({
                'Feature 1': df_corr.columns[i],
                'Feature 2': df_corr.columns[j],
                'Correlation': corr_value
            })

df_strong_corr = pd.DataFrame(strong_corr)
df_strong_corr = df_strong_corr.sort_values('Correlation', key=abs, ascending=False)

print(f"\n📊 STRONG CORRELATIONS (|r| > 0.7):")
print("=" * 80)
if len(df_strong_corr) > 0:
    for idx, row in df_strong_corr.head(15).iterrows():
        print(f"   {row['Feature 1']:35s} ↔ {row['Feature 2']:35s} : {row['Correlation']:6.3f}")
else:
    print("   ✓ No strong correlations found (good for model!)")

# --- ANALYSIS: MODERATE CORRELATIONS ---
moderate_corr = []
for i in range(len(df_corr.columns)):
    for j in range(i+1, len(df_corr.columns)):
        corr_value = df_corr.iloc[i, j]
        if 0.4 <= abs(corr_value) <= 0.7:
            moderate_corr.append({
                'Feature 1': df_corr.columns[i],
                'Feature 2': df_corr.columns[j],
                'Correlation': corr_value
            })

df_moderate_corr = pd.DataFrame(moderate_corr)
df_moderate_corr = df_moderate_corr.sort_values('Correlation', key=abs, ascending=False)

print(f"\n📊 MODERATE CORRELATIONS (0.4 < |r| < 0.7):")
print("=" * 80)
if len(df_moderate_corr) > 0:
    for idx, row in df_moderate_corr.head(15).iterrows():
        print(f"   {row['Feature 1']:35s} ↔ {row['Feature 2']:35s} : {row['Correlation']:6.3f}")

# --- SUMMARY ---
print("\n" + "="*80)
print("CORRELATION ANALYSIS SUMMARY")
print("="*80)
print(f"Total features analyzed: {len(available_features)}")
print(f"Strong correlations (|r| > 0.7): {len(df_strong_corr)}")
print(f"Moderate correlations (0.4 < |r| < 0.7): {len(df_moderate_corr)}")
print(f"\nGenerated visualizations:")
print(f"  1. correlation_matrix_featured_full.png (51 features)")
print(f"  2. correlation_matrix_featured_key.png (18 key features)")
print(f"  3. correlation_matrix_service_stats.png (12 service features)")
print("\n✅ CORRELATION ANALYSIS COMPLETE!")
print("="*80)
