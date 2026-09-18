"""
Text preprocessing utilities for multilingual misinformation detection.
Handles cleaning, normalization, and feature extraction.
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple


# Clickbait linguistic patterns
CLICKBAIT_PATTERNS = [
    # Emotional triggers
    r"\byou won'?t believe\b",
    r"\bshocking\b",
    r"\bOMG\b",
    r"\bblew my mind\b",
    r"\bjaw[ -]?drop(ping)?\b",
    r"\bmind[ -]?blow(ing)?\b",
    r"\bunbelievable\b",
    r"\bincredible\b",
    # Urgency / exclusivity
    r"\bbreaking\b",
    r"\burgent\b",
    r"\bbefore it'?s (too late|deleted|gone)\b",
    r"\blimited time\b",
    r"\bjust in\b",
    r"\bnow!\b",
    r"\btoday only\b",
    # List bait
    r"\b\d+\s+(reasons|ways|things|tips|facts|secrets|tricks|hacks|signs|photos|pictures)\b",
    r"\btop \d+\b",
    r"\bnumber \d+ will\b",
    r"\bthis is why\b",
    r"\bhere'?s why\b",
    r"\bhere'?s what\b",
    # Curiosity gap
    r"\bwhat happened next\b",
    r"\bthe result(s)? (is|are|will) (shocking|surprising|amazing)\b",
    r"\bno one (is talking about|knows)\b",
    r"\bsecret(s)? (they|the government|doctors|experts) (don'?t want you to know|hide)\b",
    r"\bthey don'?t want you to know\b",
    r"\bthe truth about\b",
    # Sensationalism
    r"\bgo viral\b",
    r"\beveryone is (talking|sharing|going crazy)\b",
    r"\binternet (breaks|explodes|goes wild)\b",
    r"\bthe internet (can'?t|is) (handle|freaking out)\b",
]

# Misinformation linguistic red flags
MISINFO_PATTERNS = [
    r"\bfake (news|media|science)\b",
    r"\bconspiracy\b",
    r"\bdeep state\b",
    r"\bplandemic\b",
    r"\bscientists (hide|lie|cover up)\b",
    r"\b(doctors|government|media) (don'?t want|hiding|suppressing)\b",
    r"\bcure(s)? (cancer|covid|aids|diabetes) (naturally|secretly|fast)\b",
    r"\b(miracle|instant) cure\b",
    r"\b100% (proven|effective|natural|safe)\b",
    r"\bexperts (agree|say) \w+ (causes|cures|prevents)\b",
    r"\bclinically proven\b.*\bno side effects\b",
    r"\bsheeple\b",
    r"\bwake up\b.*\bsheep\b",
    r"\bthey'?re (poisoning|controlling|spying)\b",
    r"\b(chemtrail|5g|microchip|nwo|illuminati)\b",
    r"\bdo your (own )?research\b",
    r"\bmainstream (media|science|medicine) (lies|hides)\b",
    r"\bsuppressed (by|from) (government|media|big pharma)\b",
]

# Excessive punctuation signals
EXCESSIVE_PUNCT_RE = re.compile(r"[!?]{2,}")
CAPS_WORDS_RE = re.compile(r"\b[A-Z]{3,}\b")


def clean_text(text: str) -> str:
    """Basic text cleaning: remove URLs, HTML tags, normalize whitespace."""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize unicode (e.g., fancy quotes → regular)
    text = unicodedata.normalize("NFKD", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """Lowercase + remove punctuation (for bag-of-words models)."""
    text = clean_text(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def extract_surface_features(text: str) -> Dict[str, float]:
    """
    Extract surface-level linguistic features that are language-agnostic
    or work across many languages.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    sentences = re.split(r"[.!?]+", cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]

    features: Dict[str, float] = {}

    # Length features
    features["char_count"] = len(cleaned)
    features["word_count"] = len(words)
    features["sentence_count"] = len(sentences)
    features["avg_word_length"] = (
        sum(len(w) for w in words) / len(words) if words else 0
    )
    features["avg_sentence_length"] = (
        sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    )

    # Punctuation features
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["excessive_punct"] = len(EXCESSIVE_PUNCT_RE.findall(text))
    features["ellipsis_count"] = text.count("...")

    # Capitalization features
    caps_words = CAPS_WORDS_RE.findall(cleaned)
    features["caps_word_count"] = len(caps_words)
    features["caps_ratio"] = len(caps_words) / len(words) if words else 0

    # Number features
    features["number_count"] = len(re.findall(r"\b\d+\b", text))

    # Quote features
    features["quote_count"] = text.count('"') + text.count("'")

    return features


def detect_clickbait_phrases(text: str) -> List[str]:
    """Return list of matched clickbait phrases (lowercased source text)."""
    found = []
    text_lower = text.lower()
    for pattern in CLICKBAIT_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            found.append(match.group(0).strip())
    return list(set(found))


def detect_misinfo_phrases(text: str) -> List[str]:
    """Return list of matched misinformation red-flag phrases."""
    found = []
    text_lower = text.lower()
    for pattern in MISINFO_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            found.append(match.group(0).strip())
    return list(set(found))


def highlight_suspicious_spans(text: str) -> List[Dict]:
    """
    Return a list of character span dicts for suspicious phrases,
    suitable for frontend highlighting.
    """
    spans = []
    text_lower = text.lower()
    all_patterns = [
        (p, "clickbait") for p in CLICKBAIT_PATTERNS
    ] + [
        (p, "misinfo") for p in MISINFO_PATTERNS
    ]

    for pattern, ptype in all_patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            spans.append({
                "start": match.start(),
                "end": match.end(),
                "text": text[match.start():match.end()],
                "type": ptype,
            })

    # De-duplicate overlapping spans (keep longest)
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    merged = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            merged.append(span)
            last_end = span["end"]

    return merged
