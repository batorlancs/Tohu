"""Dictionary baseline: replace every word by its most frequent accented form in the corpus.

Also saves the word-form counts (data/lexicon.tsv), reused as a Hungarian lexicon.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.chars import deaccent  # noqa: E402
from tohu.evaluate import evaluate  # noqa: E402

DATA = ROOT / "data"
WORD = re.compile(r"[^\W\d_]+")
SOURCES = {
    "fineweb_formal_000_00000.u16": 60_000_000,
    "fineweb_informal_000_00000.u16": 40_000_000,
    "subs_train.u16": 40_000_000,
    "wiki_train.u16": 20_000_000,
}


class DictRestorer:
    def __init__(self, best: dict[str, str]):
        self.best = best

    def _word(self, m):
        w = m.group()
        f = self.best.get(deaccent(w.lower()))
        if f is None:
            return w
        if w.isupper() and len(w) > 1:
            return f.upper()
        return f[0].upper() + f[1:] if w[0].isupper() else f

    def restore_many(self, texts):
        return [WORD.sub(self._word, deaccent(t)) for t in texts]


def main():
    counts: Counter[str] = Counter()
    for name, n in SOURCES.items():
        text = np.memmap(DATA / name, dtype=np.uint16, mode="r")[:n].tobytes().decode("utf-16-le", "replace")
        counts.update(w.lower() for w in WORD.findall(text))
        print(name, len(counts), flush=True)
    best: dict[str, tuple[str, int]] = {}
    for w, c in counts.items():
        k = deaccent(w)
        if k not in best or c > best[k][1]:
            best[k] = (w, c)
    with open(DATA / "lexicon.tsv", "w") as f:
        for w, c in counts.most_common():
            if c >= 2:
                f.write(f"{w}\t{c}\n")
    res = evaluate(DictRestorer({k: v[0] for k, v in best.items()}), n_lines=600)
    print(json.dumps({k: {m: round(v[m] * 100, 2) for m in ("vowel_acc", "word_acc", "sentence_acc")}
                      for k, v in res.items()}, indent=1))
    print("challenge errors:", res["challenge"]["errors"].most_common(40))


if __name__ == "__main__":
    main()
