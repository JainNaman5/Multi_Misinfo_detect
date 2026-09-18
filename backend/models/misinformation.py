"""
Misinformation detection model.
Uses TF-IDF + Logistic Regression (Demo Mode) — runs on any machine, no GPU needed.
Detects unreliable, biased, or fabricated content across multiple languages.
"""

import os
import joblib
import numpy as np
from typing import Dict, List

from utils.preprocessor import (
    normalize_text,
    extract_surface_features,
    detect_misinfo_phrases,
    detect_clickbait_phrases,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MISINFO_MODEL_PATH = os.path.join(MODEL_DIR, "misinfo_model.pkl")
MISINFO_VECTORIZER_PATH = os.path.join(MODEL_DIR, "misinfo_vectorizer.pkl")


class MisinformationDetector:
    """
    Misinformation detector using TF-IDF + Logistic Regression.
    Falls back to heuristic scoring when no model is loaded.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._loaded = False

    def load(self) -> bool:
        try:
            if os.path.exists(MISINFO_MODEL_PATH) and os.path.exists(MISINFO_VECTORIZER_PATH):
                self.model = joblib.load(MISINFO_MODEL_PATH)
                self.vectorizer = joblib.load(MISINFO_VECTORIZER_PATH)
                self._loaded = True
                return True
        except Exception as e:
            print(f"[MisinformationDetector] Failed to load model: {e}")
        return False

    def _heuristic_score(self, text: str) -> float:
        """
        Rule-based misinformation risk score (0–1).
        Works across languages via pattern matching and surface features.
        """
        score = 0.0
        feats = extract_surface_features(text)
        misinfo_phrases = detect_misinfo_phrases(text)
        clickbait_phrases = detect_clickbait_phrases(text)

        # Misinformation red-flag phrases (strong signal)
        score += min(len(misinfo_phrases) * 0.22, 0.66)

        # Clickbait co-occurrence raises misinfo suspicion
        score += min(len(clickbait_phrases) * 0.06, 0.12)

        # Emotional / all-caps language
        score += min(feats.get("caps_ratio", 0) * 0.25, 0.15)

        # Excessive punctuation (!!, ?!)
        score += min(feats.get("excessive_punct", 0) * 0.05, 0.10)

        return min(score, 1.0)

    def _get_reasoning(self, text: str, score: float) -> List[str]:
        """Generate human-readable reasoning chips for the result."""
        reasons = []
        misinfo_phrases = detect_misinfo_phrases(text)
        clickbait_phrases = detect_clickbait_phrases(text)
        feats = extract_surface_features(text)

        if misinfo_phrases:
            reasons.append(f"Contains {len(misinfo_phrases)} misinformation red-flag phrase(s)")
        if clickbait_phrases:
            reasons.append(f"Uses {len(clickbait_phrases)} sensationalist phrase(s)")
        if feats.get("caps_ratio", 0) > 0.15:
            reasons.append("Excessive use of ALL CAPS (emotional manipulation)")
        if feats.get("excessive_punct", 0) > 1:
            reasons.append("Heavy use of !! or ?! (urgency/alarm tactics)")
        if feats.get("exclamation_count", 0) > 3:
            reasons.append("High exclamation mark density")
        if not reasons:
            if score < 0.3:
                reasons.append("No significant red flags detected")
                reasons.append("Language appears measured and factual")
            else:
                reasons.append("General linguistic patterns suggest unreliability")

        return reasons[:5]

    def predict(self, text: str) -> Dict:
        """
        Analyze text for misinformation signals.

        Returns:
            {
                "label": str,          # "Reliable" | "Suspicious" | "Likely Fake"
                "confidence": float,   # 0–1
                "risk_score": float,   # 0–1 (higher = more suspicious)
                "reasoning": List[str],
                "red_flag_phrases": List[str],
            }
        """
        misinfo_phrases = detect_misinfo_phrases(text)
        heuristic = self._heuristic_score(text)

        if self._loaded:
            try:
                norm = normalize_text(text)
                tfidf_vec = self.vectorizer.transform([norm])
                feats = extract_surface_features(text)
                import scipy.sparse as sp
                surface_arr = np.array([[
                    feats.get("exclamation_count", 0),
                    feats.get("excessive_punct", 0),
                    feats.get("caps_ratio", 0),
                    feats.get("word_count", 0),
                    feats.get("number_count", 0),
                    len(misinfo_phrases),
                ]])
                combined = sp.hstack([tfidf_vec, sp.csr_matrix(surface_arr)])
                proba = self.model.predict_proba(combined)[0]
                ml_score = float(proba[1])
                # Ensemble: 65% ML + 35% heuristic
                risk_score = 0.65 * ml_score + 0.35 * heuristic
                confidence = float(max(proba))
            except Exception:
                risk_score = heuristic
                confidence = 0.60
        else:
            risk_score = heuristic
            confidence = 0.60 + min(heuristic * 0.25, 0.25)

        risk_score = round(min(risk_score, 1.0), 3)

        # Determine label
        if risk_score >= 0.65:
            label = "Likely Fake"
        elif risk_score >= 0.35:
            label = "Suspicious"
        else:
            label = "Reliable"

        reasoning = self._get_reasoning(text, risk_score)

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "risk_score": risk_score,
            "reasoning": reasoning,
            "red_flag_phrases": misinfo_phrases[:8],
        }


_detector: MisinformationDetector = None


def get_detector() -> MisinformationDetector:
    global _detector
    if _detector is None:
        _detector = MisinformationDetector()
        _detector.load()
    return _detector
