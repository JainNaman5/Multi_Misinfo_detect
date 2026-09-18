"""
Model training script for Multilingual Misinformation & Clickbait Detection.
Generates synthetic training data, trains TF-IDF + ML classifiers,
and saves serialized models to saved_models/.

Usage:
    python train.py

Outputs:
    saved_models/misinfo_model.pkl
    saved_models/misinfo_vectorizer.pkl
    saved_models/clickbait_model.pkl
    saved_models/clickbait_vectorizer.pkl
"""

import os
import sys
import random
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. SYNTHETIC TRAINING DATA GENERATION
# ──────────────────────────────────────────────

RELIABLE_TEMPLATES = [
    "Scientists publish peer-reviewed study on {topic} in {journal}.",
    "Government officials confirm {event} after thorough investigation.",
    "Researchers at {university} find evidence that {claim} under controlled conditions.",
    "According to the latest {agency} report, {statistic}.",
    "Health officials recommend {action} based on clinical evidence.",
    "A new study with {n} participants shows {finding} with statistical significance.",
    "The {organization} released its annual report on {topic}, showing {trend}.",
    "Economists at {institution} project {outcome} based on current data.",
    "Court documents reveal {event} after months of legal proceedings.",
    "The {country} government announced {policy} following parliamentary approval.",
    "Multiple independent sources confirm that {event} occurred on {date}.",
    "Laboratory tests conducted by {lab} verify that {product} meets safety standards.",
    "The World Health Organization updated its guidelines on {topic}.",
    "Analysis of {n} data points indicates a {trend} in {field}.",
    "Experts from {university} and {institution} collaborated on a study of {topic}.",
]

FAKE_TEMPLATES = [
    "SHOCKING: {claim} — The government doesn't want you to know this!",
    "They're hiding the TRUTH about {topic}! Wake up sheeple!",
    "100% PROVEN: {miracle} CURES {disease} overnight — doctors hate this!",
    "Deep state plot exposed: {conspiracy} — share before it's deleted!",
    "URGENT: {topic} is a hoax engineered by {villain} to control us!",
    "Scientists LIED about {topic} for decades — here's the real truth!",
    "{miracle} destroys {disease} in 3 days — big pharma suppressing it!",
    "Breaking: {person} arrested for {crime} — mainstream media silent!",
    "5G towers are actually designed to {conspiracy_action} — proof inside!",
    "ALERT: {vaccine} causes {disease} — thousands of unreported cases!",
    "The illuminati controls {institution} — leaked documents prove it!",
    "This {food} cures ALL cancers naturally — doctors won't tell you!",
    "Chemtrails contain {chemical} to make you {effect} — insider reveals all!",
    "EXPOSED: {celebrity} is secretly {conspiracy_role} — the evidence!",
    "NASA faked {event} — photos leaked proving it was staged in {location}!",
]

CLICKBAIT_TEMPLATES = [
    "You Won't Believe What {celebrity} Did Next — Number {n} Will Shock You!",
    "{n} Reasons Why {topic} Is Ruining Your Life (And You Don't Even Know It)",
    "This {adjective} {thing} Will Change Everything You Know About {topic}",
    "What Happened When This {person} Tried {action} Will Leave You Speechless",
    "The Internet Can't Handle This Incredible {topic} Moment — See Why!",
    "Doctors Are Freaking Out Over This One Weird Trick for {problem}",
    "Only {pct}% of People Can Get {n}/{n} on This {topic} Quiz — Can You?",
    "This Mom's {discovery} Is Going Viral and Everyone Is Talking About It",
    "{n} Things You're Doing Wrong Every Day (Number {n2} Especially)",
    "Wait Until You See What {celebrity} Looks Like Now — We're Shocked",
    "The Secret {experts} Have Been Hiding About {topic} For Years",
    "This {adjective} Photo Will Break Your Heart — But the Ending Is Perfect",
    "He Started {action} Every Day for 30 Days — The Results Are Unbelievable",
    "People Are Losing Their Minds Over This {topic} Discovery",
    "What This {animal} Did at {location} Has the Whole World Talking",
]

NON_CLICKBAIT_TEMPLATES = [
    "{topic} study published in {journal} this week.",
    "Local {organization} holds annual {event} on {date}.",
    "City council approves budget for {project}.",
    "Weather forecast: {weather} expected throughout the week.",
    "Sports update: {team} wins {n}-{n2} against {team2}.",
    "{country} elections scheduled for {date}.",
    "Market update: {index} closes {direction} by {pct}% today.",
    "{company} reports quarterly earnings of {amount}.",
    "Police investigating {incident} near {location}.",
    "New traffic regulations take effect starting {date}.",
    "University researchers publish findings on {topic}.",
    "Health ministry confirms {n} new cases of {disease} this week.",
    "{organization} launches initiative to support {cause}.",
    "Technology conference scheduled for {month} in {city}.",
    "Environmental report highlights changes in {ecosystem} since {year}.",
]

# Filler values for template substitution
FILLERS = {
    "topic": ["climate change", "nutrition", "mental health", "AI ethics", "renewable energy",
               "vaccine safety", "drug resistance", "social media", "quantum computing", "water quality"],
    "journal": ["Nature", "The Lancet", "JAMA", "Science", "NEJM", "Cell"],
    "event": ["the summit", "the merger", "the election", "the outbreak", "the treaty"],
    "university": ["MIT", "Harvard", "Oxford", "Stanford", "Cambridge", "IIT"],
    "claim": ["moderate exercise improves memory", "plant-based diets reduce inflammation",
               "sleep quality affects cognitive function", "air pollution increases risk of heart disease"],
    "agency": ["WHO", "CDC", "IPCC", "FDA", "UNICEF", "Eurostat"],
    "statistic": ["carbon emissions fell by 3% last year", "obesity rates have stabilized",
                  "literacy rates improved by 12% over a decade"],
    "action": ["washing hands frequently", "exercising 30 minutes daily", "wearing sunscreen"],
    "n": ["42", "100", "500", "1200", "3", "7", "10", "15", "21"],
    "finding": ["a 20% reduction in risk", "no significant correlation", "improved outcomes"],
    "organization": ["WHO", "UNICEF", "Red Cross", "Amnesty International", "Greenpeace"],
    "trend": ["a steady increase", "a notable decline", "no significant change"],
    "institution": ["World Bank", "IMF", "OECD", "Federal Reserve"],
    "country": ["France", "India", "Brazil", "Japan", "Germany"],
    "policy": ["new health guidelines", "economic stimulus package", "environmental regulations"],
    "lab": ["the National Health Laboratory", "BioTech Research Center", "GenLab Inc."],
    "product": ["the new vaccine batch", "imported food products", "medical devices"],
    "field": ["renewable energy adoption", "antibiotic resistance", "digital literacy"],
    "date": ["Monday", "next Tuesday", "this Friday", "Q4 2025"],
    "miracle": ["garlic", "turmeric", "alkaline water", "colloidal silver", "essential oils"],
    "disease": ["cancer", "diabetes", "COVID-19", "Alzheimer's", "hypertension"],
    "conspiracy": ["population control via food", "mind control via 5G", "digital ID implants"],
    "villain": ["Bill Gates", "the WHO", "big pharma", "the UN", "George Soros"],
    "person": ["whistleblower", "insider", "former agent", "anonymous doctor"],
    "crime": ["treason", "corruption", "child trafficking", "fraud"],
    "conspiracy_action": ["track your location", "control your thoughts", "sterilize the population"],
    "vaccine": ["this new shot", "the mRNA vaccine", "the flu shot"],
    "chemical": ["barium", "strontium", "aluminum nanoparticles"],
    "effect": ["infertile", "docile", "sick", "mind-controlled"],
    "celebrity": ["a famous actress", "a well-known politician", "this athlete", "a popular influencer"],
    "conspiracy_role": ["a lizard person", "a deep state operative", "a crisis actor"],
    "location": ["Area 51", "a Hollywood studio", "a government facility"],
    "adjective": ["simple", "weird", "bizarre", "incredible", "shocking", "mind-blowing"],
    "thing": ["trick", "food", "habit", "secret", "hack", "discovery"],
    "problem": ["belly fat", "back pain", "insomnia", "anxiety", "hair loss"],
    "pct": ["5", "10", "3", "1", "20"],
    "n2": ["5", "3", "8", "12", "20"],
    "animal": ["dog", "cat", "elephant", "parrot", "raccoon"],
    "discovery": ["weight loss secret", "money-saving trick", "parenting hack"],
    "team": ["the home team", "the local club", "the national squad"],
    "team2": ["rivals", "the visiting team", "the defending champions"],
    "weather": ["Rain and thunderstorms", "Sunny skies with light clouds", "Snow and strong winds"],
    "direction": ["up", "down"],
    "amount": ["$2.3 billion", "$450 million", "€1.1 billion"],
    "index": ["NASDAQ", "FTSE 100", "Sensex", "Nikkei", "S&P 500"],
    "incident": ["a vehicle break-in", "a suspected arson", "a disturbance"],
    "cause": ["youth education", "homeless veterans", "reforestation", "digital inclusion"],
    "month": ["March", "June", "October", "November"],
    "city": ["Berlin", "Tokyo", "Mumbai", "Toronto", "Lagos"],
    "ecosystem": ["coral reefs", "rainforests", "Arctic ice", "wetlands"],
    "year": ["2010", "2015", "2000", "2018"],
    "project": ["new school renovation", "road expansion", "public park upgrade"],
    "company": ["TechCorp", "GlobalBank", "AeroCo", "PharmaCo", "RetailGiant"],
    "drug": ["the experimental treatment", "the new antibiotic", "the generic alternative"],
}


def _fill(template: str) -> str:
    """Fill a template string with random filler values."""
    result = template
    for key, options in FILLERS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(options), 1)
    return result


def generate_samples(templates, n=300):
    samples = []
    for _ in range(n):
        t = random.choice(templates)
        samples.append(_fill(t))
    return samples


def build_dataset():
    print("📦 Generating synthetic training data...")

    # Misinformation dataset
    reliable = generate_samples(RELIABLE_TEMPLATES, 500)
    fake = generate_samples(FAKE_TEMPLATES, 500)
    misinfo_texts = reliable + fake
    misinfo_labels = [0] * len(reliable) + [1] * len(fake)

    # Clickbait dataset
    clickbait = generate_samples(CLICKBAIT_TEMPLATES, 500)
    not_clickbait = generate_samples(NON_CLICKBAIT_TEMPLATES, 500)
    click_texts = clickbait + not_clickbait
    click_labels = [1] * len(clickbait) + [0] * len(not_clickbait)

    return (misinfo_texts, misinfo_labels), (click_texts, click_labels)


# ──────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ──────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessor import normalize_text, extract_surface_features, detect_misinfo_phrases, detect_clickbait_phrases


def build_surface_features(texts):
    rows = []
    for text in texts:
        feats = extract_surface_features(text)
        misinfo = detect_misinfo_phrases(text)
        clickbait = detect_clickbait_phrases(text)
        rows.append([
            feats.get("exclamation_count", 0),
            feats.get("question_count", 0),
            feats.get("excessive_punct", 0),
            feats.get("caps_ratio", 0),
            feats.get("number_count", 0),
            feats.get("ellipsis_count", 0),
            feats.get("word_count", 0),
            len(misinfo),
            len(clickbait),
        ])
    return np.array(rows, dtype=np.float32)


# ──────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────

def train_misinfo(texts, labels):
    print("\n🧠 Training Misinformation Classifier...")
    norm_texts = [normalize_text(t) for t in texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(norm_texts)
    surface = build_surface_features(texts)
    X = sp.hstack([tfidf, sp.csr_matrix(surface)])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Reliable", "Fake"]))

    joblib.dump(model, MISINFO_MODEL_PATH := os.path.join(MODEL_DIR, "misinfo_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "misinfo_vectorizer.pkl"))
    print(f"   💾 Saved to {MODEL_DIR}/misinfo_model.pkl")
    return model, vectorizer


def train_clickbait(texts, labels):
    print("\n🎯 Training Clickbait Detector...")
    norm_texts = [normalize_text(t) for t in texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=6000, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(norm_texts)
    surface = build_surface_features(texts)
    X = sp.hstack([tfidf, sp.csr_matrix(surface)])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(C=2.0, max_iter=500, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Not Clickbait", "Clickbait"]))

    joblib.dump(model, os.path.join(MODEL_DIR, "clickbait_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "clickbait_vectorizer.pkl"))
    print(f"   💾 Saved to {MODEL_DIR}/clickbait_model.pkl")
    return model, vectorizer


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Multilingual Misinformation Detection — Model Trainer")
    print("=" * 60)

    (misinfo_texts, misinfo_labels), (click_texts, click_labels) = build_dataset()

    print(f"\n📊 Dataset sizes:")
    print(f"   Misinformation: {len(misinfo_texts)} samples")
    print(f"   Clickbait:      {len(click_texts)} samples")

    train_misinfo(misinfo_texts, misinfo_labels)
    train_clickbait(click_texts, click_labels)

    print("\n" + "=" * 60)
    print("  ✅ Training complete! Models saved to saved_models/")
    print("  🚀 You can now run: uvicorn app:app --reload")
    print("=" * 60)
