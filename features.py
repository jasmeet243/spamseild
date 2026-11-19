"""Rule-based extras: scam tactics, URL/phone analysis (the 'different' part of SpamShield)."""
import re
from urllib.parse import urlparse

TACTICS = {
    "Urgency": ["urgent", "immediately", "now", "hurry", "last chance", "expires", "expire", "today only",
                "within 24 hours", "within 24 hrs", "act fast", "asap", "limited time", "final notice"],
    "Fear / Threat": ["blocked", "suspended", "suspend", "deactivated", "legal action", "arrest", "penalty",
                      "fine", "terminated", "unauthorized", "unauthorised", "compromised", "closed", "expired", "returned", "held"],
    "Greed / Reward": ["won", "winner", "prize", "lottery", "lakh", "crore", "cash", "reward", "free",
                       "gift", "bonus", "cashback", "jackpot", "claim", "congratulations", "offer",
                       "earn", "work from home", "no experience", "daily income", "guaranteed", "per day", "daily"],
    "Authority": ["bank", "kyc", "rbi", "sbi", "hdfc", "icici", "paytm", "income tax", "customs", "police",
                  "government", "irs", "amazon", "microsoft", "support team", "customer care", "official",
                  "parcel", "courier", "delhivery", "dtdc", "fedex", "india post", "customs"],
    "Action Bait": ["click", "link", "verify", "update", "confirm", "login", "log in", "call now", "reply",
                    "send otp", "share otp", "otp", "pin", "password", "download", "install", "whatsapp", "pay", "fee"],
}

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly", "shorturl.at", "tiny.cc"}
BAD_TLDS = {"xyz", "top", "club", "click", "work", "loan", "gq", "tk", "ml", "cf", "ga", "buzz", "online", "site"}
BRANDS = ["sbi", "hdfc", "icici", "paytm", "amazon", "paypal", "google", "apple", "netflix", "axis", "kotak"]

URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+|\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.(?:com|net|org|in|xyz|top|club|info|co|ly|click|tk|ml|site|online|work)\b[^\s]*)", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}(?!\d)")


def detect_tactics(text: str) -> dict:
    t = " " + text.lower() + " "
    out = {}
    for name, words in TACTICS.items():
        hits = sorted({w for w in words if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t)})
        out[name] = hits
    return out


def analyse_urls(text: str) -> list:
    results = []
    for raw in URL_RE.findall(text):
        raw = raw.rstrip(".,);")
        url = raw if raw.lower().startswith("http") else "http://" + raw
        host = (urlparse(url).hostname or "").lower()
        flags = []
        if host in SHORTENERS:
            flags.append("URL shortener hides real destination")
        if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
            flags.append("IP address used instead of domain")
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        if tld in BAD_TLDS:
            flags.append(f"Suspicious TLD (.{tld})")
        if host.count("-") >= 2:
            flags.append("Many hyphens in domain")
        if "@" in raw:
            flags.append("'@' trick in URL")
        if len(host) > 30:
            flags.append("Very long domain")
        if urlparse(url).scheme == "http":
            flags.append("Not HTTPS")
        for b in BRANDS:   # brand name in domain but domain is not the real one
            if b in host and not re.search(rf"(^|\.){b}\.(com|in|co\.in)$", host):
                flags.append(f"Possible lookalike of '{b}'")
                break
        results.append({"url": raw, "host": host, "flags": flags})
    return results


def extract_phones(text: str) -> list:
    return [re.sub(r"\s+", " ", m.group()).strip() for m in PHONE_RE.finditer(text)]


SAFE_HINTS = ["do not share", "don't share", "never share", "do not disclose", "not asked", "if not done by you"]


def is_legit_warning(text: str) -> bool:
    """Genuine bank/OTP SMS usually WARN the user not to share the code (and carry no link)."""
    t = text.lower()
    return any(h in t for h in SAFE_HINTS) and not analyse_urls(text)


def rule_score(text: str) -> int:
    """0-100 score from tactics + URL flags only (independent of ML model)."""
    tac = detect_tactics(text)
    s = sum(min(len(v), 3) * 6 for v in tac.values())
    for u in analyse_urls(text):
        s += 8 + 8 * len(u["flags"])
    if is_legit_warning(text):
        s = max(0, s - 25)
    return min(100, s)


SENSITIVE_RE = re.compile(r"\b(share|send|tell|give|provide|enter|forward|reply with|confirm)\b[^.!?]{0,25}\b(otp|pin|cvv|password|passcode|card number|aadhaar|aadhar)\b", re.I)


def red_flags(text: str) -> list:
    """Hard rules that should override a low ML score."""
    flags = []
    if SENSITIVE_RE.search(text) and not is_legit_warning(text):
        flags.append("Asks you to share a sensitive code/credential (OTP, PIN, CVV, password)")
    if any(len(u["flags"]) >= 3 for u in analyse_urls(text)):
        flags.append("Link has 3+ suspicious signals")
    return flags
