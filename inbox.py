"""Inbox scanner: read emails (IMAP / .eml / .mbox) and SMS exports (.xml / .csv / .txt), then rate each with SpamShield.
Nothing is stored; mailboxes are opened READ-ONLY, so emails are never marked as read, moved or deleted."""
import csv, email, imaplib, io, mailbox, os, re, tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from email import policy
from html.parser import HTMLParser
from urllib.parse import urlparse
from predict import analyse, risk_level
from preprocess import URL_RE

MAX_CHARS = 2000   # only the first part of long emails is analysed (model was trained on short SMS)


# ---------------------------------------------------------------- HTML helpers
class _HTML(HTMLParser):
    """Turns an HTML email into plain text AND collects every link as (visible text, real href)."""
    def __init__(self):
        super().__init__(); self.text, self.links = [], []; self._href, self._atext, self._skip = None, [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"): self._skip += 1
        if tag == "a": self._href, self._atext = dict(attrs).get("href"), []
        if tag in ("br", "p", "div", "tr", "li"): self.text.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style"): self._skip = max(0, self._skip - 1)
        if tag == "a" and self._href is not None:
            self.links.append((" ".join("".join(self._atext).split()), self._href)); self._href = None
    def handle_data(self, d):
        if self._skip: return
        self.text.append(d)
        if self._href is not None: self._atext.append(d)


def html_to_text(html_str):
    p = _HTML(); p.feed(html_str)
    return re.sub(r"\n\s*\n+", "\n", "".join(p.text)).strip(), p.links


def _host(u):
    u = u if "://" in u else "http://" + u
    return (urlparse(u).hostname or "").lower()

def _base(h):
    parts = h.split("."); return ".".join(parts[-2:]) if len(parts) >= 2 else h


def link_mismatches(links):
    """Classic phishing trick: the link TEXT shows one site (paypal.com) but the real href goes elsewhere."""
    out = []
    for text, href in links:
        m = re.search(r"(?:https?://)?([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)", text.lower())
        if m and href.lower().startswith(("http", "www")):
            shown, real = _host(m.group(1)), _host(href)
            if shown and real and _base(shown) != _base(real):
                out.append(f"Link text shows '{m.group(1)}' but really goes to '{real}'")
    return out


# ---------------------------------------------------------------- parsers (all return a list of message dicts)
def _msg(source, sender, subject, date, body, links=()):
    return {"source": source, "sender": sender or "", "subject": subject or "", "date": date or "", "body": body or "", "links": list(links)}


def parse_raw_email(raw: bytes):
    m = email.message_from_bytes(raw, policy=policy.default)
    body, links = "", []
    try:
        for part in m.walk():                                   # collect real links from any HTML part
            if part.get_content_type() == "text/html":
                _, l = html_to_text(part.get_content()); links += l
        part = m.get_body(preferencelist=("plain", "html"))
        if part is not None:
            content = part.get_content()
            body = html_to_text(content)[0] if part.get_content_type() == "text/html" else content
    except Exception:
        body = str(m.get_payload())[:MAX_CHARS]
    return _msg("Email", str(m.get("from", "")), str(m.get("subject", "")), str(m.get("date", ""))[:25], body, links)


def parse_eml_files(files):            # files = list of (name, bytes)
    return [parse_raw_email(b) for _, b in files]


def parse_mbox(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mbox") as f: f.write(data); path = f.name
    try: return [parse_raw_email(m.as_bytes()) for m in mailbox.mbox(path)]
    finally: os.remove(path)


def parse_sms_xml(data: bytes):        # format of the "SMS Backup & Restore" Android app
    out = []
    for s in ET.fromstring(data).iter("sms"):
        if s.get("type") == "2": continue                      # 2 = sent by you
        try: d = datetime.fromtimestamp(int(s.get("date")) / 1000).strftime("%Y-%m-%d %H:%M")
        except Exception: d = s.get("readable_date", "")
        out.append(_msg("SMS", s.get("address"), "", d, s.get("body")))
    return out


def parse_csv(data: bytes):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8", errors="ignore"))))
    if not rows: return []
    head = [h.strip().lower() for h in rows[0]]
    pick = lambda names: next((head.index(n) for n in names if n in head), None)
    t, s, d = pick(["body", "message", "text", "sms", "content"]), pick(["address", "from", "sender", "number"]), pick(["date", "time", "timestamp"])
    if t is None: t, body_rows = 0, rows                       # no header: first column is the message
    else: body_rows = rows[1:]
    g = lambda r, i: r[i] if i is not None and i < len(r) else ""
    return [_msg("SMS", g(r, s), "", g(r, d), g(r, t)) for r in body_rows if g(r, t).strip()]


def parse_txt(data: bytes):            # one message per line
    return [_msg("SMS", "", "", "", ln.strip()) for ln in data.decode("utf-8", errors="ignore").splitlines() if ln.strip()]


def parse_uploads(files):
    """files = list of (filename, bytes). Picks the right parser from the extension."""
    out, eml = [], []
    for name, data in files:
        ext = name.lower().rsplit(".", 1)[-1]
        if ext == "eml": eml.append((name, data))
        elif ext == "mbox": out += parse_mbox(data)
        elif ext == "xml": out += parse_sms_xml(data)
        elif ext == "csv": out += parse_csv(data)
        elif ext == "txt": out += parse_txt(data)
    return out + parse_eml_files(eml)


def fetch_imap(host, user, password, folder="INBOX", limit=20, unseen_only=False):
    """Read the newest `limit` emails. Folder is opened read-only."""
    conn = imaplib.IMAP4_SSL(host)
    try:
        conn.login(user, password)
        conn.select(f'"{folder}"', readonly=True)
        _, data = conn.search(None, "UNSEEN" if unseen_only else "ALL")
        ids = data[0].split()[-limit:][::-1]                   # newest first
        out = []
        for i in ids:
            _, d = conn.fetch(i, "(BODY.PEEK[])")
            raw = next((x[1] for x in d if isinstance(x, tuple)), None)
            if raw: out.append(parse_raw_email(raw))
        return out
    finally:
        try: conn.logout()
        except Exception: pass


# ---------------------------------------------------------------- scanning
def scan(messages):
    rows = []
    for m in messages:
        text = (m["subject"] + ". " + m["body"]).strip(". ")[:MAX_CHARS]
        hrefs = " ".join(h for _, h in m["links"])[:500]
        # Emails: in SMS data a link almost always means spam, but in email links are normal.
        # So for emails the ML model reads only the WORDS (links removed); link rules judge the links.
        ml_text = URL_RE.sub(" ", text) if m["source"] == "Email" else None
        r = analyse(text + " " + hrefs, ml_text=ml_text)
        mism = link_mismatches(m["links"])
        if mism:
            r["red_flags"] = list(r["red_flags"]) + mism
            r["risk"] = max(r["risk"], 60.0); r["level"] = risk_level(r["risk"])
        reason = (r["red_flags"] or [""])[0]
        if not reason:
            t = [k for k, v in r["tactics"].items() if v]
            reason = ("Tactics: " + ", ".join(t)) if t else ("Suspicious link" if any(u["flags"] for u in r["urls"]) else "")
        rows.append({"msg": m, "result": r, "reason": reason, "preview": (m["subject"] or m["body"]).replace("\n", " ")[:70]})
    return sorted(rows, key=lambda x: -x["result"]["risk"])
