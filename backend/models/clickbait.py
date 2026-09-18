"""
Clickbait detection model.
Uses TF-IDF + Gradient Boosting, trained on synthetic + public data.
Detects sensational, curiosity-baiting, or emotionally manipulative headlines.
"""

import os
import re
import joblib
import numpy as np
from typing import Dict, List, Tuple

from utils.preprocessor import (
    normalize_text,
    extract_surface_features,
    detect_clickbait_phrases,
    EXCESSIVE_PUNCT_RE,
    CAPS_WORDS_RE,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
CLICKBAIT_MODEL_PATH = os.path.join(MODEL_DIR, "clickbait_model.pkl")
CLICKBAIT_VECTORIZER_PATH = os.path.join(MODEL_DIR, "clickbait_vectorizer.pkl")


class ClickbaitDetector:
    """
    Clickbait detector using TF-IDF features + surface heuristics
    fed into a Gradient Boosting classifier.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._loaded = False

    def load(self) -> bool:
        """Load pre-trained model and vectorizer from disk."""
        try:
            if os.path.exists(CLICKBAIT_MODEL_PATH) and os.path.exists(CLICKBAIT_VECTORIZER_PATH):
                self.model = joblib.load(CLICKBAIT_MODEL_PATH)
                self.vectorizer = joblib.load(CLICKBAIT_VECTORIZER_PATH)
                self._loaded = True
                return True
        except Exception as e:
            print(f"[ClickbaitDetector] Failed to load model: {e}")
        return False

    def _heuristic_score(self, text: str) -> float:
        """
        Rule-based clickbait scoring (0–1) as a fallback or ensemble member.
        Works across languages via pattern matching and surface features.
        """
        score = 0.0
        feats = extract_surface_features(text)
        matched_phrases = detect_clickbait_phrases(text)

        # Pattern match contribution
        score += min(len(matched_phrases) * 0.18, 0.54)

        # Excessive punctuation
        score += min(feats.get("excessive_punct", 0) * 0.08, 0.16)

        # Caps words
        score += min(feats.get("caps_ratio", 0) * 0.3, 0.15)

        # Number in headline (e.g. "10 reasons")
        if feats.get("number_count", 0) > 0:
            score += 0.07

        # Ellipsis (curiosity gap)
        score += min(feats.get("ellipsis_count", 0) * 0.05, 0.10)

        # Exclamation mark overuse
        score += min(feats.get("exclamation_count", 0) * 0.05, 0.15)

        return min(score, 1.0)

    def predict(self, text: str) -> Dict:
        """
        Analyze text for clickbait signals.

        Returns:
            {
                "is_clickbait": bool,
                "score": float (0–1),
                "label": str,
                "detected_patterns": List[str],
                "confidence": float,
            }
        """
        matched_phrases = detect_clickbait_phrases(text)
        heuristic = self._heuristic_score(text)

        if self._loaded:
            try:
                norm = normalize_text(text)
                tfidf_vec = self.vectorizer.transform([norm])
                surface = extract_surface_features(text)
                surface_arr = np.array([[
                    surface.get("exclamation_count", 0),
                    surface.get("question_count", 0),
                    surface.get("excessive_punct", 0),
                    surface.get("caps_ratio", 0),
                    surface.get("number_count", 0),
                    surface.get("ellipsis_count", 0),
                    surface.get("word_count", 0),
                ]])
                # Combine TF-IDF and surface features
                import scipy.sparse as sp
                combined = sp.hstack([tfidf_vec, sp.csr_matrix(surface_arr)])
                proba = self.model.predict_proba(combined)[0]
                ml_score = float(proba[1])
                # Ensemble: 60% ML, 40% heuristic
                score = 0.6 * ml_score + 0.4 * heuristic
                confidence = float(max(proba))
            except Exception:
                score = heuristic
                confidence = 0.65
        else:
            score = heuristic
            confidence = 0.65 + min(heuristic * 0.2, 0.2)

        score = round(min(score, 1.0), 3)
        is_clickbait = score >= 0.45

        if score >= 0.75:
            label = "High Clickbait"
        elif score >= 0.45:
            label = "Likely Clickbait"
        elif score >= 0.25:
            label = "Mild Clickbait"
        else:
            label = "Not Clickbait"

        return {
            "is_clickbait": is_clickbait,
            "score": score,
            "label": label,
            "detected_patterns": matched_phrases[:8],
            "confidence": round(confidence, 3),
        }


# Singleton instance
_detector: ClickbaitDetector = None


def get_detector() -> ClickbaitDetector:
    global _detector
    if _detector is None:
        _detector = ClickbaitDetector()
        _detector.load()
    return _detector
