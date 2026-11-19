"""Single entry point used by the app: analyse(text) -> dict with everything SpamShield knows."""
import re, joblib, numpy as np
from preprocess import clean, normalise
from features import detect_tactics, analyse_urls, extract_phones, rule_score, is_legit_warning, red_flags

_bundle = joblib.load("models/spamshield.joblib")
VEC, MODEL = _bundle["vectorizer"], _bundle["model"]
_VOCAB = VEC.vocabulary_
W_ML, W_RULE = 0.7, 0.3          # risk = 70% ML probability + 30% rule score


def word_contributions(text: str):
    """Explainability: for each original word, the Logistic-Regression coefficient of its (stemmed) unigram.
    Positive = pushes towards SPAM, negative = pushes towards HAM."""
    out = []
    for w in re.findall(r"\S+", text):
        stem = clean(w)
        c = float(MODEL.coef_[0][_VOCAB[stem]]) if stem in _VOCAB else 0.0
        out.append((w, c))
    return out


def risk_level(score: float) -> str:
    return "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"


def analyse(text: str, ml_text: str = None) -> dict:
    """ml_text: optional different text for the ML model (e.g. emails with links removed); rules always see `text`."""
    X = VEC.transform([clean(ml_text if ml_text is not None else text)])
    ml_prob = float(MODEL.predict_proba(X)[0, 1])
    tactics = detect_tactics(text)
    urls = analyse_urls(text)
    rs = rule_score(text)
    legit = is_legit_warning(text)
    # legit-style OTP warnings: trust the ML score less (dataset has no Indian bank SMS)
    risk = round((0.4 if legit else W_ML) * ml_prob * 100 + W_RULE * rs, 1)
    # hard red flags (dangerous link, request for OTP/PIN) should never be rated LOW
    flags = red_flags(text)
    if flags:
        risk = max(risk, 60.0)
    contrib = word_contributions(text)
    top = sorted({(w.strip(".,!?;:()\"'").lower(), round(c, 2)) for w, c in contrib if c > 0.8}, key=lambda x: -x[1])[:8]
    return {"ml_prob": ml_prob, "rule_score": rs, "risk": min(100.0, risk), "level": risk_level(risk),
            "tactics": tactics, "urls": urls, "phones": extract_phones(text),
            "legit_warning": legit, "red_flags": flags, "contrib": contrib, "top_words": top, "cleaned": clean(text)}
