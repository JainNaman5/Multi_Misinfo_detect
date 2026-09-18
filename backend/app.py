"""
FastAPI backend for Multilingual Misinformation & Clickbait Detection Engine.

Endpoints:
  POST /analyze              — Main analysis endpoint
  GET  /health               — Health check
  GET  /supported-languages  — List of supported languages
  GET  /stats                — Aggregate statistics

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the backend directory to sys.path so models can import utils
sys.path.insert(0, os.path.dirname(__file__))

from models.language_detect import detect_language, get_supported_languages
from models.misinformation import get_detector as get_misinfo_detector
from models.clickbait import get_detector as get_clickbait_detector
from utils.preprocessor import clean_text, highlight_suspicious_spans

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────

app = FastAPI(
    title="Multilingual Misinformation & Clickbait Detection API",
    description="Detect misinformation and clickbait across 12+ languages using NLP.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# IN-MEMORY STATS (reset on restart)
# ──────────────────────────────────────────────

_stats = {
    "total_analyzed": 0,
    "reliable_count": 0,
    "suspicious_count": 0,
    "fake_count": 0,
    "clickbait_count": 0,
    "language_counts": {},
    "recent": [],  # last 20 results
}


def _update_stats(result: dict):
    _stats["total_analyzed"] += 1
    label = result["misinformation"]["label"]
    if label == "Reliable":
        _stats["reliable_count"] += 1
    elif label == "Suspicious":
        _stats["suspicious_count"] += 1
    else:
        _stats["fake_count"] += 1

    if result["clickbait"]["is_clickbait"]:
        _stats["clickbait_count"] += 1

    lang = result["language"]["code"]
    _stats["language_counts"][lang] = _stats["language_counts"].get(lang, 0) + 1

    # Keep last 20 recent
    recent_item = {
        "text_preview": result["text_preview"],
        "language": result["language"],
        "misinfo_label": label,
        "misinfo_score": result["misinformation"]["risk_score"],
        "clickbait_label": result["clickbait"]["label"],
        "clickbait_score": result["clickbait"]["score"],
        "timestamp": result["timestamp"],
    }
    _stats["recent"].insert(0, recent_item)
    _stats["recent"] = _stats["recent"][:20]


# ──────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10000, description="Text to analyze")
    language_override: Optional[str] = Field(
        None, description="Optional ISO 639-1 language code to skip auto-detection"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "SHOCKING: Scientists HIDE the truth about 5G towers — they're tracking you!",
                "language_override": None,
            }
        }


# ──────────────────────────────────────────────
# STARTUP: LOAD MODELS
# ──────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    print("[Startup] Loading ML models...")
    misinfo = get_misinfo_detector()
    clickbait = get_clickbait_detector()
    misinfo_loaded = misinfo._loaded
    clickbait_loaded = clickbait._loaded
    print(f"[Startup] Misinformation model loaded: {misinfo_loaded}")
    print(f"[Startup] Clickbait model loaded: {clickbait_loaded}")
    if not misinfo_loaded or not clickbait_loaded:
        print("[Startup] ⚠️  Models not found. Running in heuristic-only mode.")
        print("[Startup] ℹ️  Run 'python train.py' to train and save models.")
    else:
        print("[Startup] ✅ All models loaded successfully.")


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Returns API health status and model availability."""
    misinfo = get_misinfo_detector()
    clickbait = get_clickbait_detector()
    return {
        "status": "ok",
        "models": {
            "misinformation": "loaded" if misinfo._loaded else "heuristic-only",
            "clickbait": "loaded" if clickbait._loaded else "heuristic-only",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/supported-languages")
def supported_languages():
    """Returns list of supported languages with metadata."""
    return {"languages": get_supported_languages()}


@app.get("/stats")
def get_stats():
    """Returns aggregate analysis statistics."""
    total = max(_stats["total_analyzed"], 1)
    return {
        "total_analyzed": _stats["total_analyzed"],
        "verdicts": {
            "reliable": _stats["reliable_count"],
            "suspicious": _stats["suspicious_count"],
            "fake": _stats["fake_count"],
        },
        "clickbait_detected": _stats["clickbait_count"],
        "clickbait_rate": round(_stats["clickbait_count"] / total, 3),
        "language_distribution": _stats["language_counts"],
        "recent": _stats["recent"][:10],
    }


@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    """
    Main analysis endpoint.
    Detects language, checks for misinformation, and detects clickbait.
    """
    start_time = time.time()

    # Validate & clean input
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned = clean_text(text)

    # 1. Language detection
    lang_code, lang_conf, lang_meta = detect_language(cleaned, override=req.language_override)

    # 2. Misinformation detection
    misinfo_result = get_misinfo_detector().predict(cleaned)

    # 3. Clickbait detection
    clickbait_result = get_clickbait_detector().predict(cleaned)

    # 4. Suspicious span highlighting
    spans = highlight_suspicious_spans(cleaned)

    # 5. Overall risk assessment
    risk_score = round(
        0.6 * misinfo_result["risk_score"] + 0.4 * clickbait_result["score"], 3
    )

    if risk_score >= 0.6:
        overall_verdict = "High Risk"
        overall_color = "danger"
    elif risk_score >= 0.35:
        overall_verdict = "Moderate Risk"
        overall_color = "warning"
    else:
        overall_verdict = "Low Risk"
        overall_color = "safe"

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    result = {
        "text_preview": cleaned[:120] + ("..." if len(cleaned) > 120 else ""),
        "language": {
            "code": lang_code,
            "name": lang_meta["name"],
            "flag": lang_meta["flag"],
            "rtl": lang_meta["rtl"],
            "confidence": lang_conf,
        },
        "misinformation": misinfo_result,
        "clickbait": clickbait_result,
        "overall": {
            "verdict": overall_verdict,
            "color": overall_color,
            "risk_score": risk_score,
        },
        "highlighted_spans": spans,
        "processing_time_ms": elapsed_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _update_stats(result)
    return result
