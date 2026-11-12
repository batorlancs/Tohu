"""Training-time augmentation operating on correctly accented text.

Every transform keeps the text *correctly accented*, so labels stay valid:
  * Hunglish: swap Hungarian words for English words/phrases, optionally with a
    Hungarian suffix (meetingen, deadline-ra, callról, commitoltam). The English
    part keeps no accents, the suffix keeps its accents.
  * Slang: nagyon -> naon, hogy -> h, köszönöm -> köszi, ...
  * Chat style: all lowercase, dropped punctuation, xd / :D / haha.
  * Short inputs (a single word or two), so isolated words work too.
The *input* side may also keep some accents (users sometimes type a few).
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path

from .chars import ACCENTED_CHARS, all_variants, deaccent

DATA = Path(__file__).resolve().parents[1] / "data"

# (back-vowel form, front-vowel form)
SUFFIXES = [
    ("ban", "ben"), ("ba", "be"), ("ból", "ből"), ("ra", "re"), ("ról", "ről"), ("nál", "nél"),
    ("hoz", "hez"), ("tól", "től"), ("nak", "nek"), ("val", "vel"), ("on", "en"), ("t", "t"),
    ("ot", "et"), ("ok", "ek"), ("okat", "eket"), ("ja", "je"), ("ját", "jét"), ("jával", "jével"),
    ("jára", "jére"), ("jában", "jében"), ("ig", "ig"), ("ért", "ért"), ("ként", "ként"), ("é", "é"),
    ("os", "es"), ("osan", "esen"), ("unk", "ünk"), ("unkat", "ünket"), ("od", "ed"), ("om", "em"),
    ("omat", "emet"), ("ál", "él"), ("án", "én"), ("ás", "és"), ("ást", "ést"),
    # verbal (commitolni, updatelni, pusholtam, chatelés)
    ("olni", "elni"), ("olom", "elem"), ("olod", "eled"), ("oltam", "eltem"), ("oltad", "elted"),
    ("olta", "elte"), ("oltuk", "eltük"), ("olás", "elés"), ("olást", "elést"), ("olhatod", "elheted"),
    ("olnád", "elnéd"), ("olnám", "elném"), ("olják", "elik"), ("olva", "elve"), ("olunk", "elünk"),
    ("olj", "elj"), ("oljuk", "eljük"), ("olt", "elt"), ("olós", "elős"), ("ozás", "ezés"),
    ("álni", "álni"), ("áltam", "áltam"), ("álta", "álta"), ("álás", "álás"), ("ája", "éje"),
]

SLANG = {
    "nagyon": ["naon", "nagyon", "nagyonn", "nagyon"],
    "hogy": ["h", "hogy", "hogy"],
    "szerintem": ["sztem", "szerintem"],
    "valami": ["vmi"], "valamit": ["vmit"], "valaki": ["vki"], "valakit": ["vkit"],
    "valamikor": ["vmikor"], "valahol": ["vhol"], "valamilyen": ["vmilyen"],
    "köszönöm": ["köszi", "kösz", "köszke", "köszii"],
    "köszönjük": ["köszi", "kösz"],
    "tudom": ["tom", "tudom"],
    "vagyok": ["vok", "vagyok"],
    "szóval": ["szal", "szóval"],
    "egyébként": ["amúgy", "egyébként", "amúgyis"],
    "igen": ["ja", "igen", "aha", "jah", "igeen"],
    "oké": ["oké", "oki", "okés", "ok", "okké"],
    "szia": ["szia", "csá", "cső", "hali", "helló", "sziaa", "szió", "csumi"],
    "sziasztok": ["sziasztok", "hali", "csá", "sziasztok"],
    "akkor": ["akkor", "akk"],
    "bocsánat": ["bocsi", "bocs", "sorry", "bocsánat"],
    "barátom": ["tesó", "haver", "bro", "tesóm"],
    "rendben": ["rendben", "oké", "okés", "rendbe"],
    "mindjárt": ["mindjárt", "mindjá", "mingyá"],
    "tényleg": ["tényleg", "tényleg", "tök komolyan"],
    "nagyszerű": ["tök jó", "szuper", "királyság", "nagyszerű"],
    "nagyon jó": ["tök jó", "nagyon jó", "naon jó", "brutál jó"],
    "nem tudom": ["nemtom", "nemtudom", "nem tom", "nem tudom"],
    "mit tudom én": ["mittomén", "mit tom én"],
    "mi újság": ["mizu", "mi újság", "mivan"],
    "légy szíves": ["légyszi", "pls", "légyszíves"],
    "egy kicsit": ["egy kicsit", "kicsit", "picit"],
    "fogalmam sincs": ["fogalmam sincs", "gőzöm sincs", "lövésem sincs"],
}
_SLANG_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SLANG), key=len, reverse=True)) + r")\b", re.I)
CHAT_TOKENS = ["xd", "xD", "XD", ":D", ":)", ":(", ";)", "haha", "hahaha", "lol", "pls", "thx", "btw", "omg",
               "wtf", "idk", "jaj", "uhh", "hmm", "brr", "na", "ááá", "jajj", "hűha", "húú", "óó", "ühüm"]


@lru_cache(maxsize=1)
def english_words() -> list[str]:
    """Common English words whose spelling does NOT collide with an accented Hungarian word."""
    from wordfreq import top_n_list, zipf_frequency

    out = []
    for w in top_n_list("en", 40000):
        if not (w.isascii() and w.isalpha() and len(w) >= 3 and re.search("[aeiou]", w)):
            continue
        zw = zipf_frequency(w, "hu")
        # e.g. "hat" -> "hát", "mate" -> "Máté": ambiguous, leave out
        if any(zipf_frequency(v, "hu") >= max(2.0, zw) for v in all_variants(w)[1:]):
            continue
        out.append(w)
    return out


@lru_cache(maxsize=1)
def english_phrases() -> list[str]:
    p = DATA / "english_phrases.txt"
    return p.read_text().split("\n") if p.exists() else []


def _front(word: str) -> bool:
    vs = [c for c in word.lower() if c in "aeiou"]
    return bool(vs) and vs[-1] in "ei"


def english_token(rng: random.Random) -> str:
    w = rng.choice(english_words())
    r = rng.random()
    if r < 0.12:
        w = w.capitalize()
    elif r < 0.15:
        w = w.upper()
    if rng.random() < 0.55:
        suf = rng.choice(SUFFIXES)[1 if _front(w) else 0]
        w = w + ("-" if rng.random() < 0.3 else "") + suf
    return w


def hunglish(text: str, rng: random.Random) -> str:
    toks = text.split(" ")
    idx = [i for i, t in enumerate(toks) if t[:1].isalpha()]
    if not idx:
        return text
    for i in rng.sample(idx, k=min(len(idx), rng.choice([1, 1, 1, 2, 2, 3]))):
        m = re.match(r"^(\w+)(\W*)$", toks[i])
        tail = m.group(2) if m else ""
        if rng.random() < 0.7 or not english_phrases():
            rep = english_token(rng)
        else:
            ws = rng.choice(english_phrases()).rstrip(".!?,").split()
            n = rng.randint(1, min(6, len(ws)))
            s = rng.randint(0, len(ws) - n)
            rep = " ".join(ws[s : s + n])
        toks[i] = rep + tail
    return " ".join(toks)


def slang(text: str, rng: random.Random) -> str:
    def sub(m):
        if rng.random() < 0.5:
            return m.group(0)
        r = rng.choice(SLANG[m.group(0).lower()])
        return r.capitalize() if m.group(0)[0].isupper() else r

    return _SLANG_RE.sub(sub, text)


_PUNCT = re.compile(r"[.,;:!?\"„”“()]")


def chat(text: str, rng: random.Random) -> str:
    if rng.random() < 0.7:
        text = text.lower()
    if rng.random() < 0.5:
        text = _PUNCT.sub("", text)
    if rng.random() < 0.4:
        toks = text.split(" ")
        for _ in range(rng.randint(1, 2)):
            toks.insert(rng.randint(0, len(toks)), rng.choice(CHAT_TOKENS))
        text = " ".join(toks)
    return text


def short(text: str, rng: random.Random) -> str:
    toks = [t for t in text.split() if t]
    if not toks:
        return text
    n = rng.randint(1, min(4, len(toks)))
    s = rng.randint(0, len(toks) - n)
    return " ".join(toks[s : s + n])


def partial_input(target: str, rng: random.Random) -> str:
    """Input text where the user typed some of the accents themselves."""
    keep = rng.uniform(0.1, 0.9)
    return "".join(c if (c in ACCENTED_CHARS and rng.random() < keep) else deaccent(c) for c in target)


def augment(text: str, rng: random.Random, short_p: float = 0.08) -> tuple[str, str | None]:
    """Returns (target, input_or_None). None input = fully de-accented target."""
    if rng.random() < short_p:
        text = short(text, rng)
    if rng.random() < 0.30:
        text = hunglish(text, rng)
    if rng.random() < 0.30:
        text = slang(text, rng)
    if rng.random() < 0.25:
        text = chat(text, rng)
    inp = partial_input(text, rng) if rng.random() < 0.08 else None
    return text, inp
