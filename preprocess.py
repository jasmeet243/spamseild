"""Text preprocessing for SpamShield.
Steps: leetspeak normalisation -> URL/number tokens -> lowercase -> clean -> stopwords -> stemming
"""
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = PorterStemmer()
# keep words that carry spam signal even though they are "stopwords"
KEEP = {"free", "now", "only", "call", "you", "your", "urgent"}
STOP = set(ENGLISH_STOP_WORDS) - KEEP

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|in|xyz|top|club|info|co|ly)\b\S*)", re.I)
NUM_RE = re.compile(r"\d[\d,.]*")

# Leetspeak / obfuscation map (used by normalise())
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}


_WORD_RE = re.compile(r"[A-Za-z0-9@$]+(?:![A-Za-z0-9@$]+)*")
_SKIP_RE = re.compile(r"^\d+(am|pm|p|th|st|nd|rd|k|g|mb|gb|x)$", re.I)   # '5pm', '150p', '2nd' are real


def _fix_word(m):
    w = m.group()
    letters = sum(c.isalpha() for c in w)
    if letters < 2 or len(w) < 3 or _SKIP_RE.match(w):
        return w
    if all(c.isalpha() or c in LEET for c in w):      # mixed letters + leet chars only (no 2,6,8,9 digits)
        return "".join(LEET.get(c, c) for c in w)
    return w


def normalise(text: str) -> str:
    """Undo common obfuscation: 'fr33 pr!ze 73xt w1n' -> 'free prize text win'.
    A word is only de-leeted if it mixes letters with leet symbols, so real numbers
    ('5000', '150p', '5pm') and normal '!' ('URGENT!') are left alone."""
    text = re.sub(r"\b(?:[A-Za-z]\s){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)  # 'f r e e'
    text = re.sub(r"(?<=[A-Za-z])[._*](?=[A-Za-z])", "", text)                                      # 'fr.ee', 'f*ree'
    return _WORD_RE.sub(_fix_word, text)


def clean(text: str, use_normalise: bool = True) -> str:
    text = str(text)
    text = URL_RE.sub(" urltoken ", text)      # URLs first so normalise() cannot mangle them
    if use_normalise:
        text = normalise(text)
    text = NUM_RE.sub(lambda m: " longnum " if len(re.sub(r"\D", "", m.group())) >= 5 else " num ", text)
    text = text.replace("£", " moneysym ").replace("$", " moneysym ").replace("₹", " moneysym ")
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [stemmer.stem(t) for t in text.split() if t not in STOP and len(t) > 1]
    return " ".join(tokens)
