"""Word-form frequency lexicon: unaccented key -> accented forms with corpus counts.

Used as a prior combined with the model's probabilities:
    score(form) = model_logprob(form) + alpha * log P_lexicon(form | key)
Unambiguous words get a strong prior; ambiguous ones (meg/még) a weak one, so the
model's context decides; unknown words (Hunglish, names) are left to the model.
"""

from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

from .chars import deaccent

DEFAULT_LEXICON = Path(__file__).resolve().parents[1] / "models" / "lexicon.tsv.gz"
# Whole words incl. hyphenated compounds (deadline-ra counts as one token), but not "2010-től".
TOKEN_RE = re.compile(r"(?<![\w-])[^\W\d_]+(?:-[^\W\d_]+)*(?![\w-])")


class Lexicon:
    def __init__(self, entries: dict[str, tuple[tuple[str, int], ...]]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LEXICON) -> "Lexicon":
        entries = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                key, rest = line.rstrip("\n").split("\t", 1)
                forms = []
                for item in rest.split(" "):
                    w, c = item.rsplit(":", 1)
                    forms.append((w, int(c)))
                entries[key] = tuple(forms)
        return cls(entries)

    def candidates(self, word_lower: str, min_total: int = 0) -> tuple[dict[str, float], float]:
        """({form: log prior}, log prior of an unseen form) for a lowercase word.
        Empty dict if unknown or seen fewer than min_total times (too rare to trust)."""
        forms = self.entries.get(deaccent(word_lower))
        if not forms or sum(c for _, c in forms) < min_total:
            return {}, 0.0
        denom = sum(c for _, c in forms) + 0.5 * (len(forms) + 1)
        return {w: math.log((c + 0.5) / denom) for w, c in forms}, math.log(0.5 / denom)

    def count(self, word_lower: str) -> int:
        for w, c in self.entries.get(deaccent(word_lower), ()):
            if w == word_lower:
                return c
        return 0
