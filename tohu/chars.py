"""Character-level accent (ékezet) logic shared by training and inference.

Removing Hungarian accents never changes text length, so restoration is a
per-character classification: for every base vowel the model picks one of its
variants. Every other character passes through untouched.
"""

from __future__ import annotations

import numpy as np

# Base vowel -> its variants. The label is the index into this list.
VARIANTS: dict[str, str] = {
    "a": "aá",
    "e": "eé",
    "i": "ií",
    "o": "oóöő",
    "u": "uúüű",
}
NUM_CLASSES = 4
IGNORE = -100

# Broken encodings common in subtitles / old web pages (Latin-1 instead of Latin-2).
NORMALIZE = str.maketrans({"õ": "ő", "Õ": "Ő", "û": "ű", "Û": "Ű", "ô": "ő", "Ô": "Ő"})

ACCENTED_TO_BASE: dict[str, str] = {}
LABEL_OF: dict[str, int] = {}  # any vowel (accented or not, any case) -> label
for base, variants in VARIANTS.items():
    for idx, ch in enumerate(variants):
        for c, b in ((ch, base), (ch.upper(), base.upper())):
            ACCENTED_TO_BASE[c] = b
            LABEL_OF[c] = idx

DEACCENT = str.maketrans({k: v for k, v in ACCENTED_TO_BASE.items() if k != v})
ACCENTED_CHARS = frozenset(k for k, v in ACCENTED_TO_BASE.items() if k != v)
BASE_VOWELS = frozenset("aeiouAEIOU")
# Valid-class mask per base vowel (a/e/i only have 2 variants).
VALID_MASK = {b: np.array([i < len(VARIANTS[b.lower()]) for i in range(NUM_CLASSES)]) for b in BASE_VOWELS}


def normalize(text: str) -> str:
    return text.translate(NORMALIZE)


def deaccent(text: str) -> str:
    return text.translate(DEACCENT)


def apply_label(base_char: str, label: int) -> str:
    v = VARIANTS[base_char.lower()][label]
    return v.upper() if base_char.isupper() else v


def all_variants(word: str, limit: int = 256) -> list[str]:
    """Every accent variant of an (unaccented) word, e.g. 'kerek' -> kerek, kérek, kerék, ..."""
    out = [""]
    for ch in word:
        opts = VARIANTS.get(ch.lower())
        if opts is None:
            out = [o + ch for o in out]
        else:
            opts = opts.upper() if ch.isupper() else opts
            out = [o + c for o in out for c in opts]
        if len(out) > limit:
            return []
    return out


# ---------------------------------------------------------------- vocabulary
PAD, UNK = 0, 1
# Fixed input alphabet: everything a user plausibly types. Anything else -> UNK.
_ALPHABET = (
    " \n\t"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"
    "0123456789"
    ".,;:!?-–—_'\"„”“’‘()[]{}<>/\\|@#$%^&*+=~`…•°§€£¥¢"
    "àâäãåæçèêëìîïñòøùúýÿßčćďěľĺňřšťžźżłąęśńŕăâîșțþðœ"
    "ÀÂÄÃÅÆÇÈÊËÌÎÏÑÒØÙÝŸČĆĎĚĽĹŇŘŠŤŽŹŻŁĄĘŚŃŔĂȘȚ"
)
CHARS = list(dict.fromkeys(_ALPHABET))
VOCAB_SIZE = len(CHARS) + 2
_CHAR_TO_ID = np.full(0x10000, UNK, dtype=np.int64)
for _i, _c in enumerate(CHARS):
    _CHAR_TO_ID[ord(_c)] = _i + 2

# Per-codepoint lookup tables for vectorised label/input building.
_LABEL_TABLE = np.full(0x10000, IGNORE, dtype=np.int64)
_BASE_TABLE = np.arange(0x10000, dtype=np.uint32)
for _c, _l in LABEL_OF.items():
    _LABEL_TABLE[ord(_c)] = _l
for _c, _b in ACCENTED_TO_BASE.items():
    _BASE_TABLE[ord(_c)] = ord(_b)


def to_codepoints(text: str) -> np.ndarray:
    a = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    return np.where(a < 0x10000, a, 0xFFFD).astype(np.uint32)


def encode_input(text: str) -> np.ndarray:
    return _CHAR_TO_ID[to_codepoints(text)]


def make_example(target: str, input_text: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(input ids, labels) for an accented target. Input defaults to the fully de-accented target."""
    cp = to_codepoints(target)
    labels = _LABEL_TABLE[cp]
    inp_cp = _BASE_TABLE[cp] if input_text is None else to_codepoints(input_text)
    return _CHAR_TO_ID[inp_cp], labels
