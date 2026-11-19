"""Simulate how spammers evade filters (leetspeak, symbols, spacing)."""
import random
SUBS = {"a": ["4", "@"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["5", "$"], "l": ["1"], "t": ["7"]}
# words spammers most commonly disguise (they are exactly the words filters look for)
TRIGGERS = {"free", "win", "won", "winner", "prize", "claim", "urgent", "cash", "call", "txt", "text", "reply",
            "stop", "service", "mobile", "offer", "guaranteed", "award", "chat", "congratulations", "customer",
            "latest", "contact", "ringtone", "text", "send", "now", "love", "dating", "hot"}


def obfuscate(text: str, p: float = 0.9, seed: int = 0) -> str:
    """Disguise words: every eligible letter in a trigger word is swapped (fr33, w1n, pr!z3),
    other words are disguised with probability 0.3."""
    rng = random.Random(seed)
    out = []
    for word in text.split():
        core = "".join(c for c in word if c.isalpha()).lower()
        if len(core) >= 3 and core.isalpha():
            if core in TRIGGERS and rng.random() < p:
                word = "".join(rng.choice(SUBS[c.lower()]) if c.lower() in SUBS and rng.random() < 0.8 else c for c in word)
            elif rng.random() < 0.3:
                idx = [i for i, c in enumerate(word.lower()) if c in SUBS]
                if idx:
                    i = rng.choice(idx); word = word[:i] + rng.choice(SUBS[word[i].lower()]) + word[i + 1:]
        out.append(word)
    return " ".join(out)
