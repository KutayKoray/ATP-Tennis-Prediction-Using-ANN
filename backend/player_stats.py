"""
PLAYER STATS ENGINE
===================
Processes all historical ATP match CSVs to build cumulative player statistics.
Computes the 37-feature vector for any two players + surface, matching
the feature engineering pipeline used during model training.
"""

import numpy as np
import pandas as pd
import os
import pickle
import glob
import math
from collections import defaultdict
from datetime import datetime


# ─── Default values (matching 03_feature_engineering.py) ────────────────────
DEFAULT_ACE_RATE = 0.05
DEFAULT_DF_RATE = 0.03
DEFAULT_1ST_SERVE_PCT = 0.65
DEFAULT_1ST_SERVE_WIN_PCT = 0.70


# ─── Feature order (alphabetically sorted, matching model metadata) ─────────
FEATURE_NAMES = [
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

# Binary features (skip normalization — indices in FEATURE_NAMES)
BINARY_FEATURE_INDICES = [0, 1, 4, 16, 28, 33, 34, 35]
CONTINUOUS_FEATURE_INDICES = [i for i in range(37) if i not in BINARY_FEATURE_INDICES]


class PlayerStatsEngine:
    """
    Builds and maintains player statistics from historical ATP match data.
    Provides feature vector construction for match prediction.
    """

    def __init__(self, data_dir, cache_path=None):
        """
        Args:
            data_dir: Path to the datas/ directory containing ATP CSVs
            cache_path: Path to save/load pre-computed stats cache
        """
        self.data_dir = data_dir
        self.cache_path = cache_path or os.path.join(data_dir, '..', 'backend', 'player_stats_cache.pkl')

        # Player stats (accumulated from all matches)
        self.player_stats = {}
        # H2H stats
        self.h2h_stats = {}
        # Player info (name → {id, hand, height, dob, ...})
        self.player_info = {}
        # Name → player_id mapping
        self.name_to_id = {}
        # player_id → name mapping
        self.id_to_name = {}
        # Latest ranking points
        self.ranking_points = {}
        # All available player names (for autocomplete)
        self.player_names = []

        self._loaded = False

    def load(self):
        """Load stats from cache or build from scratch."""
        if self._loaded:
            return

        if os.path.exists(self.cache_path):
            print(f"[PlayerStats] Loading cache from {self.cache_path}...")
            self._load_cache()
            print(f"[PlayerStats] ✓ Loaded {len(self.player_stats)} players from cache")
        else:
            print(f"[PlayerStats] No cache found. Building from scratch...")
            self._build_from_scratch()
            self._save_cache()
            print(f"[PlayerStats] ✓ Built and cached stats for {len(self.player_stats)} players")

        self._loaded = True

    def _load_cache(self):
        """Load pre-computed stats from pickle cache."""
        with open(self.cache_path, 'rb') as f:
            cache = pickle.load(f)
        self.player_stats = cache['player_stats']
        self.h2h_stats = cache['h2h_stats']
        self.player_info = cache['player_info']
        self.name_to_id = cache['name_to_id']
        self.id_to_name = cache['id_to_name']
        self.ranking_points = cache['ranking_points']
        self.player_names = cache['player_names']

    def _save_cache(self):
        """Save computed stats to pickle cache."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        cache = {
            'player_stats': self.player_stats,
            'h2h_stats': self.h2h_stats,
            'player_info': self.player_info,
            'name_to_id': self.name_to_id,
            'id_to_name': self.id_to_name,
            'ranking_points': self.ranking_points,
            'player_names': self.player_names,
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"[PlayerStats] ✓ Cache saved to {self.cache_path}")

    def _build_from_scratch(self):
        """Build player stats by processing all historical ATP match CSVs."""
        # 1. Load player info (height, hand, DOB)
        self._load_player_info()
        # 2. Load ranking points
        self._load_rankings()
        # 3. Process all match files chronologically
        self._process_all_matches()

    # ─── Load player bio data ───────────────────────────────────────────────
    def _load_player_info(self):
        """Load player bio from atp_players.csv."""
        players_path = os.path.join(self.data_dir, 'atp_players.csv')
        if not os.path.exists(players_path):
            print(f"[PlayerStats] WARNING: {players_path} not found")
            return

        df = pd.read_csv(players_path)
        for _, row in df.iterrows():
            pid = row['player_id']
            first = str(row.get('name_first', '')).strip()
            last = str(row.get('name_last', '')).strip()
            full_name = f"{first} {last}".strip()

            self.player_info[pid] = {
                'name': full_name,
                'hand': row.get('hand', 'R'),
                'height': row.get('height', None),
                'dob': row.get('dob', None),
            }

            if full_name and full_name != 'nan nan':
                # Store mapping (lowercase for fuzzy matching)
                self.name_to_id[full_name.lower()] = pid
                self.id_to_name[pid] = full_name

        print(f"[PlayerStats] ✓ Loaded {len(self.player_info)} player profiles")

    # ─── Load rankings ──────────────────────────────────────────────────────
    def _load_rankings(self):
        """Load latest ranking points from ranking files."""
        ranking_files = sorted(glob.glob(os.path.join(self.data_dir, 'atp_rankings_*.csv')))

        if not ranking_files:
            print("[PlayerStats] WARNING: No ranking files found")
            return

        # Read all ranking files and keep the latest entry per player
        for rf in ranking_files:
            try:
                df = pd.read_csv(rf)
                if 'player' in df.columns and 'points' in df.columns:
                    # Get the latest date entries
                    if 'ranking_date' in df.columns:
                        latest_date = df['ranking_date'].max()
                        df_latest = df[df['ranking_date'] == latest_date]
                    else:
                        df_latest = df

                    for _, row in df_latest.iterrows():
                        pid = row['player']
                        points = row.get('points', 0)
                        if pd.notna(points):
                            self.ranking_points[pid] = int(points)
            except Exception as e:
                print(f"[PlayerStats] Warning: Error reading {rf}: {e}")

        print(f"[PlayerStats] ✓ Loaded rankings for {len(self.ranking_points)} players")

    # ─── Process all matches ────────────────────────────────────────────────
    def _process_all_matches(self):
        """Process all ATP match CSVs chronologically to build player stats."""
        match_files = sorted(glob.glob(os.path.join(self.data_dir, 'atp_matches_*.csv')))

        if not match_files:
            print("[PlayerStats] WARNING: No match files found")
            return

        # Also include the combined file if it exists
        combined = os.path.join(self.data_dir, 'all_atp_matches_1968_2024.csv')

        # Use individual year files for proper chronological processing
        total_matches = 0

        for mf in match_files:
            basename = os.path.basename(mf)
            # Skip the combined file — use individual year files instead
            if basename.startswith('all_'):
                continue

            try:
                df = pd.read_csv(mf, low_memory=False)
                # Sort by date within each file
                if 'tourney_date' in df.columns:
                    df = df.sort_values('tourney_date').reset_index(drop=True)

                self._process_match_dataframe(df)
                total_matches += len(df)
            except Exception as e:
                print(f"[PlayerStats] Warning: Error processing {basename}: {e}")

        # Build player names list for autocomplete
        self.player_names = sorted(set(
            name for name in self.id_to_name.values()
            if name and name != 'nan nan'
        ))

        print(f"[PlayerStats] ✓ Processed {total_matches:,} matches from {len(match_files)} files")

    def _process_match_dataframe(self, df):
        """Process a DataFrame of matches, updating player/h2h stats."""
        for _, row in df.iterrows():
            winner_id = row.get('winner_id')
            loser_id = row.get('loser_id')

            if pd.isna(winner_id) or pd.isna(loser_id):
                continue

            winner_id = int(winner_id)
            loser_id = int(loser_id)
            surface = str(row.get('surface', 'Hard'))

            # Also capture player names from match data (backup)
            w_name = row.get('winner_name', '')
            l_name = row.get('loser_name', '')
            if pd.notna(w_name) and str(w_name).lower() not in self.name_to_id:
                self.name_to_id[str(w_name).lower()] = winner_id
                self.id_to_name[winner_id] = str(w_name)
            if pd.notna(l_name) and str(l_name).lower() not in self.name_to_id:
                self.name_to_id[str(l_name).lower()] = loser_id
                self.id_to_name[loser_id] = str(l_name)

            # Initialize stats dicts if needed
            for pid in [winner_id, loser_id]:
                if pid not in self.player_stats:
                    self.player_stats[pid] = {
                        'matches': 0,
                        'wins': 0,
                        'losses': 0,
                        'matches_by_surface': defaultdict(int),
                        'wins_by_surface': defaultdict(int),
                        'recent_matches': [],
                        'service_history': {
                            'ace_rates': [],
                            'df_rates': [],
                            '1st_serve_pcts': [],
                            '1st_serve_win_pcts': [],
                        }
                    }

            # ── Update winner stats ──
            ws = self.player_stats[winner_id]
            ws['matches'] += 1
            ws['wins'] += 1
            ws['matches_by_surface'][surface] += 1
            ws['wins_by_surface'][surface] += 1
            ws['recent_matches'].append(1)
            if len(ws['recent_matches']) > 20:
                ws['recent_matches'].pop(0)

            # Service history (winner)
            w_svpt = row.get('w_svpt', 0)
            if pd.notna(w_svpt) and w_svpt > 0:
                w_svpt = float(w_svpt)
                ace = float(row.get('w_ace', 0)) if pd.notna(row.get('w_ace')) else 0
                df_val = float(row.get('w_df', 0)) if pd.notna(row.get('w_df')) else 0
                ws['service_history']['ace_rates'].append(ace / w_svpt)
                ws['service_history']['df_rates'].append(df_val / w_svpt)

                w_1stIn = row.get('w_1stIn', 0)
                if pd.notna(w_1stIn) and float(w_1stIn) > 0:
                    w_1stIn = float(w_1stIn)
                    w_1stWon = float(row.get('w_1stWon', 0)) if pd.notna(row.get('w_1stWon')) else 0
                    ws['service_history']['1st_serve_pcts'].append(w_1stIn / w_svpt)
                    ws['service_history']['1st_serve_win_pcts'].append(w_1stWon / w_1stIn)

                for key in ws['service_history']:
                    if len(ws['service_history'][key]) > 50:
                        ws['service_history'][key].pop(0)

            # ── Update loser stats ──
            ls = self.player_stats[loser_id]
            ls['matches'] += 1
            ls['losses'] += 1
            ls['matches_by_surface'][surface] += 1
            ls['recent_matches'].append(0)
            if len(ls['recent_matches']) > 20:
                ls['recent_matches'].pop(0)

            # Service history (loser)
            l_svpt = row.get('l_svpt', 0)
            if pd.notna(l_svpt) and l_svpt > 0:
                l_svpt = float(l_svpt)
                ace = float(row.get('l_ace', 0)) if pd.notna(row.get('l_ace')) else 0
                df_val = float(row.get('l_df', 0)) if pd.notna(row.get('l_df')) else 0
                ls['service_history']['ace_rates'].append(ace / l_svpt)
                ls['service_history']['df_rates'].append(df_val / l_svpt)

                l_1stIn = row.get('l_1stIn', 0)
                if pd.notna(l_1stIn) and float(l_1stIn) > 0:
                    l_1stIn = float(l_1stIn)
                    l_1stWon = float(row.get('l_1stWon', 0)) if pd.notna(row.get('l_1stWon')) else 0
                    ls['service_history']['1st_serve_pcts'].append(l_1stIn / l_svpt)
                    ls['service_history']['1st_serve_win_pcts'].append(l_1stWon / l_1stIn)

                for key in ls['service_history']:
                    if len(ls['service_history'][key]) > 50:
                        ls['service_history'][key].pop(0)

            # ── Update H2H stats ──
            matchup = tuple(sorted([winner_id, loser_id]))
            if matchup not in self.h2h_stats:
                self.h2h_stats[matchup] = {'matches': 0, 'player1_wins': 0}

            self.h2h_stats[matchup]['matches'] += 1
            if matchup[0] == winner_id:
                self.h2h_stats[matchup]['player1_wins'] += 1

    # ─── Player lookup ──────────────────────────────────────────────────────
    def find_player_id(self, name):
        """
        Find player ID by name (case-insensitive, supports partial matching).
        Returns (player_id, full_name) or (None, None) if not found.
        """
        name_lower = name.strip().lower()

        # Exact match
        if name_lower in self.name_to_id:
            pid = self.name_to_id[name_lower]
            return pid, self.id_to_name.get(pid, name)

        # Partial match (name contains query)
        matches = []
        for stored_name, pid in self.name_to_id.items():
            if name_lower in stored_name:
                matches.append((pid, self.id_to_name.get(pid, stored_name)))

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Return the closest match (shortest name that contains query)
            matches.sort(key=lambda x: len(x[1]))
            return matches[0]

        return None, None

    # ─── Get player features ────────────────────────────────────────────────
    def _get_player_features(self, player_id, surface):
        """Get feature values for a single player."""
        stats = self.player_stats.get(player_id, None)
        info = self.player_info.get(player_id, {})

        # Age computation
        dob = info.get('dob', None)
        age = 27.0  # default
        if dob and pd.notna(dob):
            try:
                dob_str = str(int(dob))
                birth_year = int(dob_str[:4])
                birth_month = int(dob_str[4:6]) if len(dob_str) >= 6 else 6
                now = datetime.now()
                age = now.year - birth_year + (now.month - birth_month) / 12.0
            except (ValueError, TypeError):
                age = 27.0

        # Height
        height = info.get('height', None)
        if height is None or pd.isna(height):
            height = 185.0  # default
        else:
            height = float(height)

        # Hand
        hand = str(info.get('hand', 'R'))
        is_lefty = 1 if hand == 'L' else 0

        # Rank points
        rank_points = self.ranking_points.get(player_id, 100)

        # Experience
        experience = max(0, age - 18.0)

        # Stats-derived features
        if stats and stats['matches'] > 0:
            career_win_rate = stats['wins'] / stats['matches']
            surface_matches = stats['matches_by_surface'].get(surface, 0)
            if surface_matches > 0:
                surface_win_rate = stats['wins_by_surface'].get(surface, 0) / surface_matches
            else:
                surface_win_rate = 0.0

            recent = stats['recent_matches'][-10:]
            recent_form = sum(recent) / len(recent) if recent else 0.0

            sh = stats['service_history']
            avg_ace_rate = np.mean(sh['ace_rates'][-20:]) if sh['ace_rates'] else DEFAULT_ACE_RATE
            avg_df_rate = np.mean(sh['df_rates'][-20:]) if sh['df_rates'] else DEFAULT_DF_RATE
            avg_1st_pct = np.mean(sh['1st_serve_pcts'][-20:]) if sh['1st_serve_pcts'] else DEFAULT_1ST_SERVE_PCT
            avg_1st_win = np.mean(sh['1st_serve_win_pcts'][-20:]) if sh['1st_serve_win_pcts'] else DEFAULT_1ST_SERVE_WIN_PCT
        else:
            career_win_rate = 0.0
            surface_win_rate = 0.0
            recent_form = 0.0
            avg_ace_rate = DEFAULT_ACE_RATE
            avg_df_rate = DEFAULT_DF_RATE
            avg_1st_pct = DEFAULT_1ST_SERVE_PCT
            avg_1st_win = DEFAULT_1ST_SERVE_WIN_PCT

        return {
            'age': age,
            'avg_1st_serve_pct': avg_1st_pct,
            'avg_1st_serve_win_pct': avg_1st_win,
            'avg_ace_rate': avg_ace_rate,
            'avg_df_rate': avg_df_rate,
            'career_win_rate': career_win_rate,
            'experience': experience,
            'ht': height,
            'is_lefty': is_lefty,
            'rank_points': rank_points,
            'recent_form': recent_form,
            'surface_win_rate': surface_win_rate,
        }

    # ─── Build the 37-feature vector ────────────────────────────────────────
    def build_feature_vector(self, player1_name, player2_name, surface):
        """
        Build the 37-element feature vector for a match prediction.

        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            surface: "Hard", "Clay", or "Grass"

        Returns:
            dict with keys:
              - features: numpy array of shape (37,)
              - p1_id, p2_id: player IDs
              - p1_name, p2_name: resolved full names
              - error: error message if any, else None
        """
        # Look up players
        p1_id, p1_full = self.find_player_id(player1_name)
        p2_id, p2_full = self.find_player_id(player2_name)

        if p1_id is None:
            return {'error': f"Player not found: '{player1_name}'", 'features': None}
        if p2_id is None:
            return {'error': f"Player not found: '{player2_name}'", 'features': None}

        # Get per-player features
        p1 = self._get_player_features(p1_id, surface)
        p2 = self._get_player_features(p2_id, surface)

        # Hand matchup
        both_left = 1 if (p1['is_lefty'] == 1 and p2['is_lefty'] == 1) else 0
        both_right = 1 if (p1['is_lefty'] == 0 and p2['is_lefty'] == 0) else 0
        mixed = 1 if (p1['is_lefty'] != p2['is_lefty']) else 0

        # H2H
        matchup = tuple(sorted([p1_id, p2_id]))
        h2h = self.h2h_stats.get(matchup, {'matches': 0, 'player1_wins': 0})
        h2h_matches = h2h['matches']
        if h2h_matches > 0:
            if matchup[0] == p1_id:
                h2h_win_rate = h2h['player1_wins'] / h2h_matches
            else:
                h2h_win_rate = 1 - (h2h['player1_wins'] / h2h_matches)
        else:
            h2h_win_rate = 0.5

        # Temporal features (current date)
        now = datetime.now()
        month = now.month
        quarter = (month - 1) // 3 + 1
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        # Surface one-hot
        surface_lower = surface.lower().strip()
        surface_clay = 1 if surface_lower == 'clay' else 0
        surface_grass = 1 if surface_lower == 'grass' else 0
        surface_hard = 1 if surface_lower == 'hard' else 0

        # Tournament importance (default to ATP 500 = 2)
        tourney_importance = 2

        # ── Assemble in alphabetical order (matching FEATURE_NAMES) ──
        features = np.array([
            both_left,                      # 0  both_left_handed
            both_right,                     # 1  both_right_handed
            h2h_matches,                    # 2  h2h_matches
            h2h_win_rate,                   # 3  h2h_win_rate
            mixed,                          # 4  mixed_handed
            month,                          # 5  month
            month_cos,                      # 6  month_cos
            month_sin,                      # 7  month_sin
            p1['age'],                      # 8  p1_age
            p1['avg_1st_serve_pct'],        # 9  p1_avg_1st_serve_pct
            p1['avg_1st_serve_win_pct'],    # 10 p1_avg_1st_serve_win_pct
            p1['avg_ace_rate'],             # 11 p1_avg_ace_rate
            p1['avg_df_rate'],              # 12 p1_avg_df_rate
            p1['career_win_rate'],          # 13 p1_career_win_rate
            p1['experience'],              # 14 p1_experience
            p1['ht'],                       # 15 p1_ht
            p1['is_lefty'],                 # 16 p1_is_lefty
            p1['rank_points'],              # 17 p1_rank_points
            p1['recent_form'],              # 18 p1_recent_form
            p1['surface_win_rate'],         # 19 p1_surface_win_rate
            p2['age'],                      # 20 p2_age
            p2['avg_1st_serve_pct'],        # 21 p2_avg_1st_serve_pct
            p2['avg_1st_serve_win_pct'],    # 22 p2_avg_1st_serve_win_pct
            p2['avg_ace_rate'],             # 23 p2_avg_ace_rate
            p2['avg_df_rate'],              # 24 p2_avg_df_rate
            p2['career_win_rate'],          # 25 p2_career_win_rate
            p2['experience'],              # 26 p2_experience
            p2['ht'],                       # 27 p2_ht
            p2['is_lefty'],                 # 28 p2_is_lefty
            p2['rank_points'],              # 29 p2_rank_points
            p2['recent_form'],              # 30 p2_recent_form
            p2['surface_win_rate'],         # 31 p2_surface_win_rate
            quarter,                        # 32 quarter
            surface_clay,                   # 33 surface_clay
            surface_grass,                  # 34 surface_grass
            surface_hard,                   # 35 surface_hard
            tourney_importance,             # 36 tourney_importance
        ], dtype=np.float64)

        return {
            'features': features,
            'p1_id': p1_id,
            'p2_id': p2_id,
            'p1_name': p1_full,
            'p2_name': p2_full,
            'p1_stats': p1,
            'p2_stats': p2,
            'h2h': {'matches': h2h_matches, 'p1_win_rate': h2h_win_rate},
            'error': None,
        }

    def get_player_names(self):
        """Return sorted list of all available player names."""
        return self.player_names

    def search_players(self, query, limit=20):
        """Search player names by partial match."""
        query_lower = query.strip().lower()
        if not query_lower:
            return []
        results = [
            name for name in self.player_names
            if query_lower in name.lower()
        ]
        return results[:limit]
