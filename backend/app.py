"""
FLASK API SERVER
================
REST API for tennis match prediction.

Endpoints:
  POST /api/predict  — Predict match outcome
  GET  /api/players  — Search player names
  GET  /api/health   — Health check
"""

import os
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from player_stats import PlayerStatsEngine
from model import TennisPredictor

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'datas')
MODEL_DIR = os.path.join(PROJECT_DIR, 'saved_models')
CACHE_PATH = os.path.join(BASE_DIR, 'player_stats_cache.pkl')

# ── Initialize Flask ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# ── Lazy initialization (avoids gunicorn preload crash) ──────────────────────
predictor = None
stats_engine = None
init_error = None
init_done = False

def ensure_initialized():
    """Initialize model and stats on first request (not at import time)."""
    global predictor, stats_engine, init_error, init_done
    if init_done:
        return init_error is None

    try:
        print("=" * 60)
        print("🎾 ATP TENNIS MATCH PREDICTION API")
        print("=" * 60)

        print("\n[1/2] Loading model...")
        predictor = TennisPredictor(MODEL_DIR)
        predictor.load()

        print("\n[2/2] Loading player statistics...")
        stats_engine = PlayerStatsEngine(DATA_DIR, cache_path=CACHE_PATH)
        stats_engine.load()

        print("\n" + "=" * 60)
        print("✅ API READY")
        print("=" * 60)
        init_done = True
        return True
    except Exception as e:
        init_error = str(e)
        init_done = True
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        traceback.print_exc()
        return False


# ── Root endpoint ────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def root():
    """Root endpoint — shows if API is running."""
    return jsonify({
        'service': 'Tennis Match Prediction API',
        'status': 'ready' if (init_done and init_error is None) else 'starting',
        'endpoints': ['/api/health', '/api/players?q=name', '/api/predict'],
    })


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if not ensure_initialized():
        return jsonify({'status': 'error', 'error': init_error}), 503
    return jsonify({
        'status': 'ok',
        'model': predictor.metadata.get('model_type', 'unknown'),
        'activation': predictor.metadata.get('activation', 'unknown'),
        'players_loaded': len(stats_engine.player_names),
    })


@app.route('/api/players', methods=['GET'])
def search_players():
    """
    Search for player names.
    Query params:
      - q: search query (partial name match)
      - limit: max results (default 20)
    """
    if not ensure_initialized():
        return jsonify({'error': 'Server is still initializing, please wait...'}), 503

    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 20))

    if not query:
        return jsonify({'players': []})

    results = stats_engine.search_players(query, limit=limit)
    return jsonify({'players': results})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict match outcome.

    Request body (JSON):
    {
      "player1": "Novak Djokovic",
      "player2": "Rafael Nadal",
      "surface": "Clay"
    }

    Response:
    {
      "player1": { "name": "Novak Djokovic", "win_probability": 0.42, "stats": {...} },
      "player2": { "name": "Rafael Nadal",   "win_probability": 0.58, "stats": {...} },
      "predicted_winner": "Rafael Nadal",
      "confidence": 58.12,
      "surface": "Clay",
      "h2h": { "matches": 59, "p1_win_rate": 0.475 }
    }
    """
    data = request.get_json()

    if not ensure_initialized():
        return jsonify({'error': 'Server is still initializing, please wait...'}), 503

    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    player1 = data.get('player1', '').strip()
    player2 = data.get('player2', '').strip()
    surface = data.get('surface', '').strip()

    # Validate inputs
    if not player1:
        return jsonify({'error': 'player1 is required'}), 400
    if not player2:
        return jsonify({'error': 'player2 is required'}), 400
    if not surface:
        return jsonify({'error': 'surface is required'}), 400

    surface_cap = surface.capitalize()
    if surface_cap not in ['Hard', 'Clay', 'Grass']:
        return jsonify({'error': f"Invalid surface: '{surface}'. Must be Hard, Clay, or Grass"}), 400

    # Build feature vector
    try:
        result = stats_engine.build_feature_vector(player1, player2, surface_cap)

        if result['error']:
            return jsonify({'error': result['error']}), 404

        # Run prediction
        prediction = predictor.predict_match(result['features'])

        # Build response
        response = {
            'player1': {
                'name': result['p1_name'],
                'win_probability': prediction['p1_win_probability'],
                'stats': {
                    'age': round(result['p1_stats']['age'], 1),
                    'height': result['p1_stats']['ht'],
                    'hand': 'Left' if result['p1_stats']['is_lefty'] else 'Right',
                    'rank_points': result['p1_stats']['rank_points'],
                    'career_win_rate': round(result['p1_stats']['career_win_rate'] * 100, 1),
                    'surface_win_rate': round(result['p1_stats']['surface_win_rate'] * 100, 1),
                    'recent_form': round(result['p1_stats']['recent_form'] * 100, 1),
                    'avg_ace_rate': round(result['p1_stats']['avg_ace_rate'] * 100, 2),
                    'avg_1st_serve_pct': round(result['p1_stats']['avg_1st_serve_pct'] * 100, 1),
                },
            },
            'player2': {
                'name': result['p2_name'],
                'win_probability': prediction['p2_win_probability'],
                'stats': {
                    'age': round(result['p2_stats']['age'], 1),
                    'height': result['p2_stats']['ht'],
                    'hand': 'Left' if result['p2_stats']['is_lefty'] else 'Right',
                    'rank_points': result['p2_stats']['rank_points'],
                    'career_win_rate': round(result['p2_stats']['career_win_rate'] * 100, 1),
                    'surface_win_rate': round(result['p2_stats']['surface_win_rate'] * 100, 1),
                    'recent_form': round(result['p2_stats']['recent_form'] * 100, 1),
                    'avg_ace_rate': round(result['p2_stats']['avg_ace_rate'] * 100, 2),
                    'avg_1st_serve_pct': round(result['p2_stats']['avg_1st_serve_pct'] * 100, 1),
                },
            },
            'predicted_winner': result['p1_name'] if prediction['predicted_winner'] == 1 else result['p2_name'],
            'confidence': prediction['confidence'],
            'surface': surface_cap,
            'h2h': result['h2h'],
        }

        return jsonify(response)

    except Exception as e:
        print(f"[Predict] Error: {e}")
        return jsonify({'error': 'Something went wrong while processing the prediction. Please try again.'}), 500


# ── Run server ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"\n🚀 Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
