"""
MODEL INFERENCE MODULE
======================
Loads the saved ANN model weights and performs forward propagation
for match prediction. Matches the notebook architecture exactly:
  37 → 128 → 64 → 1  (LeakyReLU + Sigmoid)
"""

import numpy as np
import json
import os

from player_stats import CONTINUOUS_FEATURE_INDICES, BINARY_FEATURE_INDICES


class TennisPredictor:
    """
    Loads saved model weights, scaler, and metadata.
    Performs inference on a 37-feature input vector.
    """

    def __init__(self, model_dir):
        """
        Args:
            model_dir: Path to saved_models/ directory containing
                       best_model_weights.npz, best_model_scaler.npz,
                       best_model_metadata.json
        """
        self.model_dir = model_dir
        self.parameters = {}
        self.scaler_mean = None
        self.scaler_scale = None
        self.metadata = {}
        self._loaded = False

    def load(self):
        """Load model weights, scaler, and metadata."""
        if self._loaded:
            return

        # 1. Load weights
        weights_path = os.path.join(self.model_dir, 'best_model_weights.npz')
        weights = np.load(weights_path)
        self.parameters = {
            'W1': weights['W1'],
            'b1': weights['b1'],
            'W2': weights['W2'],
            'b2': weights['b2'],
            'W3': weights['W3'],
            'b3': weights['b3'],
        }
        print(f"[Model] ✓ Loaded weights: W1{self.parameters['W1'].shape}, "
              f"W2{self.parameters['W2'].shape}, W3{self.parameters['W3'].shape}")

        # 2. Load scaler
        scaler_path = os.path.join(self.model_dir, 'best_model_scaler.npz')
        scaler = np.load(scaler_path)
        self.scaler_mean = scaler['mean']
        self.scaler_scale = scaler['scale']
        print(f"[Model] ✓ Loaded scaler: mean({len(self.scaler_mean)}), "
              f"scale({len(self.scaler_scale)})")

        # 3. Load metadata
        metadata_path = os.path.join(self.model_dir, 'best_model_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"[Model] ✓ Loaded metadata: {self.metadata['activation']} "
              f"{self.metadata['architecture']}")

        self._loaded = True

    # ── Activation functions (matching notebook exactly) ─────────────────
    @staticmethod
    def sigmoid(Z):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-Z))

    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        """Leaky ReLU activation function."""
        return np.maximum(alpha * Z, Z)

    # ── Forward propagation ──────────────────────────────────────────────
    def forward_propagation(self, X):
        """
        Forward propagation through the 2-hidden-layer network.
        X: input column vector (37, 1) or (37, m)
        Returns: A3 (prediction probability)
        """
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        W3 = self.parameters['W3']
        b3 = self.parameters['b3']

        # Layer 1: Input → H1
        Z1 = np.dot(W1, X) + b1
        A1 = self.leaky_relu(Z1)

        # Layer 2: H1 → H2
        Z2 = np.dot(W2, A1) + b2
        A2 = self.leaky_relu(Z2)

        # Layer 3: H2 → Output
        Z3 = np.dot(W3, A2) + b3
        A3 = self.sigmoid(Z3)

        return A3

    # ── Normalize features ───────────────────────────────────────────────
    def normalize_features(self, features):
        """
        Apply StandardScaler normalization to continuous features only.
        Binary features (surface, hand, etc.) are left as-is.

        Args:
            features: numpy array of shape (37,) — raw feature values

        Returns:
            normalized features array of shape (37,)
        """
        normalized = features.copy()

        # Only normalize continuous features
        continuous_values = features[CONTINUOUS_FEATURE_INDICES]
        normalized_values = (continuous_values - self.scaler_mean) / self.scaler_scale
        normalized[CONTINUOUS_FEATURE_INDICES] = normalized_values

        return normalized

    # ── Predict match ────────────────────────────────────────────────────
    def predict_match(self, features):
        """
        Make a prediction from raw (unnormalized) features.

        Args:
            features: numpy array of shape (37,) — raw feature values

        Returns:
            dict with:
              - p1_win_probability: float (0-1)
              - p2_win_probability: float (0-1)
              - predicted_winner: 1 or 2
              - confidence: float (0-100)
        """
        # 1. Normalize
        normalized = self.normalize_features(features)

        # 2. Reshape to column vector (37, 1)
        X = normalized.reshape(-1, 1)

        # 3. Forward propagation
        A3 = self.forward_propagation(X)
        p1_prob = float(A3.flatten()[0])
        p2_prob = 1.0 - p1_prob

        # 4. Decision
        predicted_winner = 1 if p1_prob > 0.5 else 2
        confidence = max(p1_prob, p2_prob) * 100

        return {
            'p1_win_probability': round(p1_prob, 4),
            'p2_win_probability': round(p2_prob, 4),
            'predicted_winner': predicted_winner,
            'confidence': round(confidence, 2),
        }
