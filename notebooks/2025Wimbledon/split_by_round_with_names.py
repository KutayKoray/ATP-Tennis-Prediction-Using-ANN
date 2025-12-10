"""
SPLIT WIMBLEDON DATASET BY ROUND (WITH PLAYER NAMES)
====================================================
This script splits the normalized Wimbledon dataset into separate CSV files
for each round, keeping player names at the END for reference.
"""

import pandas as pd
from pathlib import Path

# Paths
INPUT_FILE = Path(__file__).parent / '2025_wimbledon_normalized.csv'
OUTPUT_FOLDER = Path(__file__).parent / 'rounds'

print("="*80)
print("SPLITTING WIMBLEDON DATASET BY ROUND (WITH PLAYER NAMES)")
print("="*80)

# Create output folder
OUTPUT_FOLDER.mkdir(exist_ok=True)

# --- STEP 1: LOAD DATA ---
print("\n[STEP 1] Loading normalized Wimbledon data...")
print("-" * 80)

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df)} matches")
print(f"   Columns: {len(df.columns)}")

# --- STEP 2: IDENTIFY ROUNDS ---
print("\n[STEP 2] Identifying rounds...")
print("-" * 80)

rounds = df['round'].unique()
print(f"✓ Found {len(rounds)} rounds: {sorted(rounds)}")

# Round name mapping
ROUND_NAMES = {
    '1': 'Round1_FirstRound',
    '2': 'Round2_SecondRound',
    '3': 'Round3_ThirdRound',
    '4': 'Round4_FourthRound',
    'Q': 'QuarterFinals',
    'S': 'SemiFinals',
    'F': 'Final'
}

# --- STEP 3: ORGANIZE COLUMNS ---
print("\n[STEP 3] Organizing columns...")
print("-" * 80)

# Columns to remove (day, round - not needed)
REMOVE_COLS = ['day', 'round']

# Player name columns (keep for reference, but at the end)
NAME_COLS = ['player1_name', 'player2_name']

# Model feature columns (37 features)
MODEL_FEATURES = [
    'both_left_handed', 'both_right_handed', 'h2h_matches', 'h2h_win_rate',
    'mixed_handed', 'month', 'month_cos', 'month_sin', 'p1_age',
    'p1_avg_1st_serve_pct', 'p1_avg_1st_serve_win_pct', 'p1_avg_ace_rate',
    'p1_avg_df_rate', 'p1_career_win_rate', 'p1_experience', 'p1_ht',
    'p1_is_lefty', 'p1_rank_points', 'p1_recent_form', 'p1_surface_win_rate',
    'p2_age', 'p2_avg_1st_serve_pct', 'p2_avg_1st_serve_win_pct',
    'p2_avg_ace_rate', 'p2_avg_df_rate', 'p2_career_win_rate', 'p2_experience',
    'p2_ht', 'p2_is_lefty', 'p2_rank_points', 'p2_recent_form',
    'p2_surface_win_rate', 'quarter', 'surface_clay', 'surface_grass',
    'surface_hard', 'tourney_importance'
]

# Column order: MODEL_FEATURES + winner + split + player names
COLUMN_ORDER = MODEL_FEATURES + ['winner', 'split'] + NAME_COLS

print(f"✓ Column organization:")
print(f"   - Model features: {len(MODEL_FEATURES)} (first)")
print(f"   - Winner + split: 2 (middle)")
print(f"   - Player names: 2 (last, for reference)")
print(f"   - Total: {len(COLUMN_ORDER)} columns")

# --- STEP 4: SPLIT BY ROUND ---
print("\n[STEP 4] Splitting by round and saving...")
print("-" * 80)

round_stats = []

for round_code in sorted(rounds):
    # Filter matches for this round
    round_df = df[df['round'] == round_code].copy()
    
    # Reorder columns: features first, then winner/split, then names
    round_df_ordered = round_df[COLUMN_ORDER]
    
    # Get round name
    round_name = ROUND_NAMES.get(str(round_code), f'Round_{round_code}')
    
    # Save to CSV
    output_file = OUTPUT_FOLDER / f'{round_name}.csv'
    round_df_ordered.to_csv(output_file, index=False)
    
    # Store stats
    round_stats.append({
        'round_code': round_code,
        'round_name': round_name,
        'matches': len(round_df_ordered),
        'file': output_file.name
    })
    
    print(f"✓ {round_name:25s} - {len(round_df_ordered):3d} matches → {output_file.name}")

# --- STEP 5: SUMMARY ---
print("\n[STEP 5] Summary...")
print("-" * 80)

print(f"\n📊 Created {len(round_stats)} round files:")
print(f"\n{'Round':<25s} {'Matches':<10s} {'File':<30s}")
print("-" * 70)
for stat in round_stats:
    print(f"{stat['round_name']:<25s} {stat['matches']:<10d} {stat['file']:<30s}")

print(f"\n📁 All files saved to: {OUTPUT_FOLDER}")
print(f"\n✅ Each file contains:")
print(f"   - {len(MODEL_FEATURES)} model features (columns 1-37)")
print(f"   - winner, split (columns 38-39)")
print(f"   - player1_name, player2_name (columns 40-41, for reference)")

# --- STEP 6: VERIFY ONE FILE ---
print("\n[STEP 6] Verification (checking Final)...")
print("-" * 80)

final_file = OUTPUT_FOLDER / 'Final.csv'
if final_file.exists():
    verify_df = pd.read_csv(final_file)
    
    print(f"\n📋 File: {final_file.name}")
    print(f"   Rows: {len(verify_df)}")
    print(f"   Columns: {len(verify_df.columns)}")
    
    print(f"\n   Column structure:")
    print(f"   - Columns 1-37: Model features")
    print(f"   - Column 38: winner")
    print(f"   - Column 39: split")
    print(f"   - Column 40: player1_name")
    print(f"   - Column 41: player2_name")
    
    print(f"\n   Final match:")
    print(f"   - Player 1: {verify_df.iloc[0]['player1_name']}")
    print(f"   - Player 2: {verify_df.iloc[0]['player2_name']}")
    print(f"   - Winner: Player {verify_df.iloc[0]['winner']}")
    print(f"   - p1_rank_points (normalized): {verify_df.iloc[0]['p1_rank_points']:.4f}")
    print(f"   - p2_rank_points (normalized): {verify_df.iloc[0]['p2_rank_points']:.4f}")

print("\n" + "="*80)
print("✅ SPLITTING COMPLETE!")
print("="*80)
print(f"\n📁 Output folder: {OUTPUT_FOLDER}")
print(f"🎯 {len(round_stats)} round files ready!")
print(f"\n💡 Usage:")
print(f"   - For model prediction: Use columns 1-37 (features)")
print(f"   - For result interpretation: Use columns 40-41 (player names)")
