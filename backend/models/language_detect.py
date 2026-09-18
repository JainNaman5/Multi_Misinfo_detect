"""
Language detection module using an ensemble of langdetect and langid
for robust multilingual text identification.
"""

import re
from typing import Tuple, Optional

try:
    from langdetect import detect as ld_detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 42  # Make detection deterministic
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import langid
    langid.set_languages(["en", "hi", "fr", "es", "de", "ar", "zh", "pt", "ru", "ja", "ko", "it"])
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False


# Supported language metadata
SUPPORTED_LANGUAGES = {
    "en": {"name": "English",    "flag": "🇺🇸", "rtl": False},
    "hi": {"name": "Hindi",      "flag": "🇮🇳", "rtl": False},
    "fr": {"name": "French",     "flag": "🇫🇷", "rtl": False},
    "es": {"name": "Spanish",    "flag": "🇪🇸", "rtl": False},
    "de": {"name": "German",     "flag": "🇩🇪", "rtl": False},
    "ar": {"name": "Arabic",     "flag": "🇸🇦", "rtl": True},
    "zh": {"name": "Chinese",    "flag": "🇨🇳", "rtl": False},
    "pt": {"name": "Portuguese", "flag": "🇧🇷", "rtl": False},
    "ru": {"name": "Russian",    "flag": "🇷🇺", "rtl": False},
    "ja": {"name": "Japanese",   "flag": "🇯🇵", "rtl": False},
    "ko": {"name": "Korean",     "flag": "🇰🇷", "rtl": False},
    "it": {"name": "Italian",    "flag": "🇮🇹", "rtl": False},
}


def _is_arabic_script(text: str) -> bool:
    arabic_re = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")
    return bool(arabic_re.search(text))


def _is_devanagari_script(text: str) -> bool:
    deva_re = re.compile(r"[\u0900-\u097F]+")
    return bool(deva_re.search(text))


def _is_cjk_script(text: str) -> bool:
    cjk_re = re.compile(r"[\u4e00-\u9fff\u3040-\u30FF\uAC00-\uD7AF]+")
    return bool(cjk_re.search(text))


def _script_based_hint(text: str) -> Optional[str]:
    """Fast script-based language hint before running statistical models."""
    if _is_arabic_script(text):
        return "ar"
    if _is_devanagari_script(text):
        return "hi"
    if _is_cjk_script(text):
        return "zh"  # Could be ja or ko — let statistical model refine
    return None


def detect_language(text: str, override: Optional[str] = None) -> Tuple[str, float, dict]:
    """
    Detect language of text using ensemble method.

    Returns:
        (lang_code, confidence, meta_dict)
        where meta_dict has keys: name, flag, rtl
    """
    if override and override in SUPPORTED_LANGUAGES:
        meta = SUPPORTED_LANGUAGES[override]
        return override, 1.0, meta

    if not text or len(text.strip()) < 5:
        return "en", 0.5, SUPPORTED_LANGUAGES["en"]

    # Fast script-based detection
    script_hint = _script_based_hint(text)

    lang_code = "en"
    confidence = 0.5
    votes = {}

    # langdetect vote
    if LANGDETECT_AVAILABLE:
        try:
            ld_lang = ld_detect(text)
            # Normalize zh variants
            if ld_lang.startswith("zh"):
                ld_lang = "zh"
            if ld_lang in SUPPORTED_LANGUAGES:
                votes[ld_lang] = votes.get(ld_lang, 0) + 2  # weight = 2
        except LangDetectException:
            pass

    # langid vote
    if LANGID_AVAILABLE:
        try:
            lid_lang, lid_conf = langid.classify(text)
            if lid_lang.startswith("zh"):
                lid_lang = "zh"
            if lid_lang in SUPPORTED_LANGUAGES:
                votes[lid_lang] = votes.get(lid_lang, 0) + 1
        except Exception:
            pass

    # Script hint vote (strong signal)
    if script_hint:
        votes[script_hint] = votes.get(script_hint, 0) + 3

    if votes:
        lang_code = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[lang_code] / total
    else:
        lang_code = "en"
        confidence = 0.4

    meta = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])
    return lang_code, round(confidence, 3), meta


def get_supported_languages() -> list:
    """Return list of all supported languages for the API."""
    return [
        {"code": code, **meta}
        for code, meta in SUPPORTED_LANGUAGES.items()
    ]
