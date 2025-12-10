"""
NORMALIZE 2025 WIMBLEDON DATASET
=================================
This script normalizes the Wimbledon dataset using the same scaler
that was used for training data, while preserving metadata columns.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
WIMBLEDON_CSV = Path(__file__).parent / '2025_wimbledon_featured.csv'
SCALER_FILE = Path(__file__).parent.parent / 'FeatureEngineering' / 'feature_scaler.pkl'
OUTPUT_FILE = Path(__file__).parent / '2025_wimbledon_normalized.csv'

print("="*80)
print("NORMALIZING 2025 WIMBLEDON DATASET")
print("="*80)

# --- STEP 1: LOAD WIMBLEDON DATA ---
print("\n[STEP 1] Loading Wimbledon data...")
print("-" * 80)

wimbledon_df = pd.read_csv(WIMBLEDON_CSV)
print(f"✓ Loaded {len(wimbledon_df)} matches")
print(f"   Columns: {len(wimbledon_df.columns)}")

# --- STEP 2: LOAD SCALER ---
print("\n[STEP 2] Loading scaler from training...")
print("-" * 80)

if not SCALER_FILE.exists():
    print(f"❌ ERROR: Scaler file not found at {SCALER_FILE}")
    exit(1)

scaler_info = joblib.load(SCALER_FILE)
scaler = scaler_info['scaler']
features_to_normalize_idx = scaler_info['features_to_normalize_idx']
features_to_skip_idx = scaler_info['features_to_skip_idx']
feature_names = scaler_info['feature_names']

print(f"✓ Loaded scaler")
print(f"   Features to normalize: {len(features_to_normalize_idx)}")
print(f"   Features to skip (binary/one-hot): {len(features_to_skip_idx)}")
print(f"   Total features: {len(feature_names)}")

# --- STEP 3: PREPARE DATA ---
print("\n[STEP 3] Preparing data for normalization...")
print("-" * 80)

# Metadata columns to preserve
METADATA_COLS = ['day', 'round', 'player1_name', 'player2_name']

# Check if all required features exist
missing_features = [f for f in feature_names if f not in wimbledon_df.columns]
if missing_features:
    print(f"❌ ERROR: Missing features in Wimbledon data:")
    for feat in missing_features:
        print(f"   - {feat}")
    exit(1)

print(f"✓ All required features present")

# Extract feature values in the correct order
X_wimbledon = wimbledon_df[feature_names].values
print(f"   Feature matrix shape: {X_wimbledon.shape}")

# --- STEP 4: NORMALIZE ---
print("\n[STEP 4] Normalizing features...")
print("-" * 80)

# Create a copy for normalized data
X_normalized = X_wimbledon.copy()

# Only normalize the continuous features (same as training)
if features_to_normalize_idx:
    X_normalized[:, features_to_normalize_idx] = scaler.transform(
        X_wimbledon[:, features_to_normalize_idx]
    )
    
    print(f"✓ Normalized {len(features_to_normalize_idx)} continuous features")
    print(f"   Normalized features statistics:")
    print(f"   - Mean: {X_normalized[:, features_to_normalize_idx].mean():.6f} (should be ~0)")
    print(f"   - Std: {X_normalized[:, features_to_normalize_idx].std():.6f} (should be ~1)")
    print(f"   - Min: {X_normalized[:, features_to_normalize_idx].min():.3f}")
    print(f"   - Max: {X_normalized[:, features_to_normalize_idx].max():.3f}")

# Verify binary features are still 0 or 1
if features_to_skip_idx:
    binary_values = X_normalized[:, features_to_skip_idx]
    print(f"\n   Binary/One-hot features (should be 0 or 1):")
    print(f"   - Unique values: {np.unique(binary_values)}")
    print(f"   - Min: {binary_values.min():.1f}, Max: {binary_values.max():.1f}")

# --- STEP 5: CREATE NORMALIZED DATAFRAME ---
print("\n[STEP 5] Creating normalized dataset...")
print("-" * 80)

# Create dataframe with normalized features
normalized_df = pd.DataFrame(X_normalized, columns=feature_names)

# Add metadata columns at the beginning
for col in METADATA_COLS:
    if col in wimbledon_df.columns:
        normalized_df.insert(0, col, wimbledon_df[col].values)

# Add winner column (not normalized, just copied)
if 'winner' in wimbledon_df.columns:
    normalized_df['winner'] = wimbledon_df['winner'].values

# Add split column
if 'split' in wimbledon_df.columns:
    normalized_df['split'] = wimbledon_df['split'].values

print(f"✓ Created normalized dataframe")
print(f"   Rows: {len(normalized_df)}")
print(f"   Columns: {len(normalized_df.columns)}")

# --- STEP 6: SAVE ---
print("\n[STEP 6] Saving normalized dataset...")
print("-" * 80)

normalized_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")

# --- STEP 7: VERIFICATION ---
print("\n[STEP 7] Verification...")
print("-" * 80)

print(f"\n📊 Sample comparison (first match):")
print(f"\n   Original values:")
print(f"   - p1_rank_points: {wimbledon_df.iloc[0]['p1_rank_points']:.1f}")
print(f"   - p1_age: {wimbledon_df.iloc[0]['p1_age']:.1f}")
print(f"   - surface_grass: {wimbledon_df.iloc[0]['surface_grass']:.1f}")

print(f"\n   Normalized values:")
print(f"   - p1_rank_points: {normalized_df.iloc[0]['p1_rank_points']:.4f}")
print(f"   - p1_age: {normalized_df.iloc[0]['p1_age']:.4f}")
print(f"   - surface_grass: {normalized_df.iloc[0]['surface_grass']:.1f} (unchanged - binary)")

print(f"\n   Metadata (preserved):")
print(f"   - day: {normalized_df.iloc[0]['day']}")
print(f"   - round: {normalized_df.iloc[0]['round']}")
print(f"   - player1_name: {normalized_df.iloc[0]['player1_name']}")
print(f"   - player2_name: {normalized_df.iloc[0]['player2_name']}")
print(f"   - winner: {normalized_df.iloc[0]['winner']}")

print("\n" + "="*80)
print("✅ NORMALIZATION COMPLETE!")
print("="*80)
print(f"\n📁 Output file: {OUTPUT_FILE}")
print(f"🎯 Ready for model prediction!")
print(f"\nNote: Binary features (surface, hand matchup) were NOT normalized")
print(f"      They remain as 0 or 1, just like in training data")
