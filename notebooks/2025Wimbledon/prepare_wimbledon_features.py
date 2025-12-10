"""
PREPARE 2025 WIMBLEDON DATASET WITH FEATURES
=============================================
This script creates a feature-engineered dataset for 2025 Wimbledon matches
using historical ATP data and the match results from Day1-14.json files.

Steps:
1. Load 2025 Wimbledon matches from JSON files
2. Load historical ATP data (1968-2024)
3. Calculate features for each player at the time of their match
4. Create final dataset ready for model prediction
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
WIMBLEDON_FOLDER = Path(__file__).parent
HISTORICAL_DATA = WIMBLEDON_FOLDER.parent.parent / 'datas' / 'all_atp_matches_1968_2024.csv'
OUTPUT_FILE = WIMBLEDON_FOLDER / '2025_wimbledon_featured.csv'

# Wimbledon 2025 dates
WIMBLEDON_START_DATE = '2025-06-30'
WIMBLEDON_END_DATE = '2025-07-13'

print("="*80)
print("PREPARING 2025 WIMBLEDON DATASET WITH FEATURES")
print("="*80)

# --- STEP 1: LOAD WIMBLEDON MATCHES ---
print("\n[STEP 1] Loading 2025 Wimbledon matches...")
print("-" * 80)

def load_wimbledon_matches():
    """Load all matches from Day1-14.json files"""
    all_matches = []
    
    for day_num in range(1, 15):
        json_file = WIMBLEDON_FOLDER / f'Day{day_num}.json'
        
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for match in data.get('matches', []):
                # Extract basic match info
                match_info = {
                    'day': day_num,
                    'round': match.get('roundCode', ''),
                    'round_name': match.get('roundName', ''),
                    'player1_name': f"{match['team1']['firstNameA']} {match['team1']['lastNameA']}".strip(),
                    'player2_name': f"{match['team2']['firstNameA']} {match['team2']['lastNameA']}".strip(),
                    'player1_id': match['team1'].get('idA', ''),
                    'player2_id': match['team2'].get('idA', ''),
                    'winner': int(match.get('winner', 0)),
                    'surface': 'Grass',
                    'tourney_level': 'G',  # Grand Slam
                    'match_date': WIMBLEDON_START_DATE  # Approximate
                }
                all_matches.append(match_info)
    
    return pd.DataFrame(all_matches)

wimbledon_df = load_wimbledon_matches()
print(f"✓ Loaded {len(wimbledon_df)} matches")
print(f"   Rounds: {wimbledon_df['round'].unique()}")
print(f"   Unique players: {len(set(wimbledon_df['player1_name']) | set(wimbledon_df['player2_name']))}")

# --- STEP 2: LOAD HISTORICAL DATA ---
print("\n[STEP 2] Loading historical ATP data...")
print("-" * 80)

if not HISTORICAL_DATA.exists():
    print(f"❌ ERROR: Historical data not found at {HISTORICAL_DATA}")
    print(f"   Please ensure the file exists.")
    exit(1)

hist_df = pd.read_csv(HISTORICAL_DATA)
print(f"✓ Loaded {len(hist_df)} historical matches")
print(f"   Date range: {hist_df['tourney_date'].min()} to {hist_df['tourney_date'].max()}")

# Convert date to datetime
hist_df['tourney_date'] = pd.to_datetime(hist_df['tourney_date'], format='%Y%m%d', errors='coerce')

# Filter: Only matches BEFORE Wimbledon 2025
hist_df = hist_df[hist_df['tourney_date'] < WIMBLEDON_START_DATE].copy()
print(f"✓ Filtered to {len(hist_df)} matches before Wimbledon 2025")

# --- STEP 3: CALCULATE PLAYER FEATURES ---
print("\n[STEP 3] Calculating player features from historical data...")
print("-" * 80)

def calculate_player_features(player_name, player_id, reference_date):
    """
    Calculate all features for a player based on their history before reference_date
    """
    # Filter matches where this player participated (before reference date)
    player_matches = hist_df[
        ((hist_df['winner_name'] == player_name) | (hist_df['loser_name'] == player_name)) &
        (hist_df['tourney_date'] < reference_date)
    ].copy()
    
    if len(player_matches) == 0:
        # New player with no history - return default values
        return {
            'rank_points': 0,
            'age': 25,  # Default age
            'ht': 185,  # Default height
            'is_lefty': 0,
            'experience': 0,
            'career_win_rate': 0.5,
            'surface_win_rate': 0.5,
            'recent_form': 0.5,
            'avg_ace_rate': 0.05,
            'avg_df_rate': 0.03,
            'avg_1st_serve_pct': 0.60,
            'avg_1st_serve_win_pct': 0.65,
        }
    
    # Get most recent match for current stats
    latest_match = player_matches.iloc[-1]
    
    # Determine if player was winner or loser in latest match
    is_winner = latest_match['winner_name'] == player_name
    
    # Basic attributes
    rank_points = latest_match['winner_rank_points'] if is_winner else latest_match['loser_rank_points']
    age = latest_match['winner_age'] if is_winner else latest_match['loser_age']
    ht = latest_match['winner_ht'] if is_winner else latest_match['loser_ht']
    hand = latest_match['winner_hand'] if is_winner else latest_match['loser_hand']
    is_lefty = 1 if hand == 'L' else 0
    
    # Handle missing values
    rank_points = rank_points if pd.notna(rank_points) else 0
    age = age if pd.notna(age) else 25
    ht = ht if pd.notna(ht) else 185
    
    # Experience (years as professional)
    first_match_date = player_matches['tourney_date'].min()
    experience = (pd.to_datetime(reference_date) - first_match_date).days / 365.25
    
    # Career win rate
    wins = (player_matches['winner_name'] == player_name).sum()
    total_matches = len(player_matches)
    career_win_rate = wins / total_matches if total_matches > 0 else 0.5
    
    # Surface-specific win rate (Grass for Wimbledon)
    grass_matches = player_matches[player_matches['surface'] == 'Grass']
    if len(grass_matches) > 0:
        grass_wins = (grass_matches['winner_name'] == player_name).sum()
        surface_win_rate = grass_wins / len(grass_matches)
    else:
        surface_win_rate = career_win_rate  # Fallback to career rate
    
    # Recent form (last 20 matches)
    recent_matches = player_matches.tail(20)
    recent_wins = (recent_matches['winner_name'] == player_name).sum()
    recent_form = recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0.5
    
    # Service statistics (last 20 matches where data available)
    recent_matches_with_stats = recent_matches.tail(20)
    
    # Calculate service stats
    if is_winner:
        aces = recent_matches_with_stats['w_ace'].fillna(0)
        dfs = recent_matches_with_stats['w_df'].fillna(0)
        svpt = recent_matches_with_stats['w_svpt'].fillna(100)
        first_in = recent_matches_with_stats['w_1stIn'].fillna(60)
        first_won = recent_matches_with_stats['w_1stWon'].fillna(40)
    else:
        aces = recent_matches_with_stats['l_ace'].fillna(0)
        dfs = recent_matches_with_stats['l_df'].fillna(0)
        svpt = recent_matches_with_stats['l_svpt'].fillna(100)
        first_in = recent_matches_with_stats['l_1stIn'].fillna(60)
        first_won = recent_matches_with_stats['l_1stWon'].fillna(40)
    
    avg_ace_rate = (aces / svpt.replace(0, 100)).mean() if len(svpt) > 0 else 0.05
    avg_df_rate = (dfs / svpt.replace(0, 100)).mean() if len(svpt) > 0 else 0.03
    avg_1st_serve_pct = (first_in / svpt.replace(0, 100)).mean() if len(svpt) > 0 else 0.60
    avg_1st_serve_win_pct = (first_won / first_in.replace(0, 60)).mean() if len(first_in) > 0 else 0.65
    
    return {
        'rank_points': float(rank_points),
        'age': float(age),
        'ht': float(ht),
        'is_lefty': int(is_lefty),
        'experience': float(experience),
        'career_win_rate': float(career_win_rate),
        'surface_win_rate': float(surface_win_rate),
        'recent_form': float(recent_form),
        'avg_ace_rate': float(avg_ace_rate),
        'avg_df_rate': float(avg_df_rate),
        'avg_1st_serve_pct': float(avg_1st_serve_pct),
        'avg_1st_serve_win_pct': float(avg_1st_serve_win_pct),
    }

def calculate_h2h(player1_name, player2_name, reference_date):
    """Calculate head-to-head statistics"""
    h2h_matches = hist_df[
        (((hist_df['winner_name'] == player1_name) & (hist_df['loser_name'] == player2_name)) |
         ((hist_df['winner_name'] == player2_name) & (hist_df['loser_name'] == player1_name))) &
        (hist_df['tourney_date'] < reference_date)
    ]
    
    h2h_count = len(h2h_matches)
    
    if h2h_count == 0:
        return 0, 0.5  # No history, assume 50-50
    
    p1_wins = ((h2h_matches['winner_name'] == player1_name)).sum()
    h2h_win_rate = p1_wins / h2h_count
    
    return h2h_count, h2h_win_rate

# Calculate features for all matches
print("Calculating features for each match...")
featured_matches = []

for idx, match in wimbledon_df.iterrows():
    if (idx + 1) % 20 == 0:
        print(f"   Processing match {idx + 1}/{len(wimbledon_df)}...")
    
    # Get player features
    p1_features = calculate_player_features(
        match['player1_name'], 
        match['player1_id'],
        match['match_date']
    )
    
    p2_features = calculate_player_features(
        match['player2_name'],
        match['player2_id'],
        match['match_date']
    )
    
    # Head-to-head
    h2h_matches, h2h_win_rate = calculate_h2h(
        match['player1_name'],
        match['player2_name'],
        match['match_date']
    )
    
    # Hand matchup
    both_right = (1 - p1_features['is_lefty']) * (1 - p2_features['is_lefty'])
    both_left = p1_features['is_lefty'] * p2_features['is_lefty']
    mixed = 1 - both_right - both_left
    
    # Temporal features
    month = 6  # June
    quarter = 2  # Q2
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Combine all features
    match_features = {
        # Match info
        'day': match['day'],
        'round': match['round'],
        'player1_name': match['player1_name'],
        'player2_name': match['player2_name'],
        'winner': match['winner'],
        
        # Player 1 features
        'p1_rank_points': p1_features['rank_points'],
        'p1_age': p1_features['age'],
        'p1_ht': p1_features['ht'],
        'p1_is_lefty': p1_features['is_lefty'],
        'p1_experience': p1_features['experience'],
        'p1_career_win_rate': p1_features['career_win_rate'],
        'p1_surface_win_rate': p1_features['surface_win_rate'],
        'p1_recent_form': p1_features['recent_form'],
        'p1_avg_ace_rate': p1_features['avg_ace_rate'],
        'p1_avg_df_rate': p1_features['avg_df_rate'],
        'p1_avg_1st_serve_pct': p1_features['avg_1st_serve_pct'],
        'p1_avg_1st_serve_win_pct': p1_features['avg_1st_serve_win_pct'],
        
        # Player 2 features
        'p2_rank_points': p2_features['rank_points'],
        'p2_age': p2_features['age'],
        'p2_ht': p2_features['ht'],
        'p2_is_lefty': p2_features['is_lefty'],
        'p2_experience': p2_features['experience'],
        'p2_career_win_rate': p2_features['career_win_rate'],
        'p2_surface_win_rate': p2_features['surface_win_rate'],
        'p2_recent_form': p2_features['recent_form'],
        'p2_avg_ace_rate': p2_features['avg_ace_rate'],
        'p2_avg_df_rate': p2_features['avg_df_rate'],
        'p2_avg_1st_serve_pct': p2_features['avg_1st_serve_pct'],
        'p2_avg_1st_serve_win_pct': p2_features['avg_1st_serve_win_pct'],
        
        # Head-to-head
        'h2h_matches': h2h_matches,
        'h2h_win_rate': h2h_win_rate,
        
        # Tournament context
        'tourney_importance': 4,  # Grand Slam
        'surface_hard': 0,
        'surface_clay': 0,
        'surface_grass': 1,
        
        # Hand matchup
        'both_right_handed': both_right,
        'both_left_handed': both_left,
        'mixed_handed': mixed,
        
        # Temporal
        'month': month,
        'quarter': quarter,
        'month_sin': month_sin,
        'month_cos': month_cos,
    }
    
    featured_matches.append(match_features)

featured_df = pd.DataFrame(featured_matches)
print(f"✓ Calculated features for {len(featured_df)} matches")

# --- STEP 4: SAVE DATASET ---
print("\n[STEP 4] Saving featured dataset...")
print("-" * 80)

featured_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"   Rows: {len(featured_df)}")
print(f"   Columns: {len(featured_df.columns)}")

# --- STEP 5: SUMMARY ---
print("\n[STEP 5] Dataset Summary")
print("-" * 80)

print(f"\n📊 Features per match: {len(featured_df.columns) - 5} (excluding metadata)")
print(f"   - Player 1 features: 12")
print(f"   - Player 2 features: 12")
print(f"   - Head-to-head: 2")
print(f"   - Tournament context: 4")
print(f"   - Hand matchup: 3")
print(f"   - Temporal: 4")

print(f"\n📋 Sample match:")
sample = featured_df.iloc[0]
print(f"   {sample['player1_name']} vs {sample['player2_name']}")
print(f"   P1 rank points: {sample['p1_rank_points']:.0f}")
print(f"   P2 rank points: {sample['p2_rank_points']:.0f}")
print(f"   P1 grass win rate: {sample['p1_surface_win_rate']:.2%}")
print(f"   P2 grass win rate: {sample['p2_surface_win_rate']:.2%}")
print(f"   Winner: Player {sample['winner']}")

print("\n" + "="*80)
print("✅ DATASET PREPARATION COMPLETE!")
print("="*80)
print(f"\n📁 Output file: {OUTPUT_FILE}")
print(f"🎯 Ready for model prediction!")
print(f"\nNext steps:")
print(f"1. Load the scaler: scaler = joblib.load('feature_scaler.pkl')")
print(f"2. Normalize features using the scaler")
print(f"3. Make predictions with your trained model")
