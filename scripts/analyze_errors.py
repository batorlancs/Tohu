"""Break down the restorer's word errors by cause, to see where accuracy is lost.

Each wrong word (gold -> output) gets one category, checked in this order:
  noisy-gold     gold is written without accents but the output form is far more common:
                 the reference text is probably wrong, not the model
  foreign        gold has no accents and is an English/foreign word the model accented
  proper-noun    capitalized word not at sentence start (names, places, brands)
  gold-oov       gold form is unknown to the lexicon (rare word, compound, typo)
  non-word       output form never occurs in the corpus while gold does: fixable by decoding
  ambiguous      both forms are real, common words (kerek/kérek, meg/még): needs context
  rare-choice    both forms exist but one is rare
Also reports the vowel confusions, the word frequency of errors, where in the word the
error is (stem vs. suffix), and accuracy on short 1-4 word inputs.

    uv run python scripts/analyze_errors.py [--model models/tohu.pt] [--lines 3000] [--lexicon ...]
    TOHU_DATA=data/clean ...   evaluates on the cleaned test sets
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.chars import deaccent  # noqa: E402
from tohu.evaluate import WORD_RE, load_eval_set, score  # noqa: E402
from tohu.restore import Restorer, _zipf  # noqa: E402

SETS = ("web-formal", "web-informal", "subtitles", "wikipedia", "challenge", "hunglish", "short")
CATS = ("noisy-gold", "foreign", "proper-noun", "gold-oov", "non-word", "ambiguous", "rare-choice")


def short_set(n: int, seed: int = 0) -> list[str]:
    """1-4 word fragments, like a single word or phrase typed on its own."""
    rng = random.Random(seed)
    base = load_eval_set("web-informal", n, seed) + load_eval_set("subtitles", n, seed)
    out = []
    for t in base:
        words = t.split()
        k = rng.randint(1, min(4, len(words)))
        i = rng.randint(0, len(words) - k)
        frag = " ".join(words[i : i + k]).strip(".,!?;:\"„”()")
        if any(deaccent(c) != c for c in frag) or rng.random() < 0.3:
            out.append(frag)
    return out[:n]


def categorize(gold: str, out: str, start: int, text: str, lex) -> str:
    g, o = gold.lower(), out.lower()
    cg, co = lex.count(g), lex.count(o)
    if g == deaccent(g) and co >= 20 * max(cg, 1) and co >= 50:
        return "noisy-gold"
    if g == deaccent(g) and _zipf(g, "en") >= 2.5 and _zipf(g, "hu") < _zipf(g, "en"):
        return "foreign"
    before = text[:start].rstrip()
    if gold[0].isupper() and before and before[-1] not in ".!?:\"„-–":
        return "proper-noun"
    if cg == 0:
        return "gold-oov"
    if co == 0:
        return "non-word"
    if min(cg, co) >= 0.05 * max(cg, co):
        return "ambiguous"
    return "rare-choice"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "tohu.pt"))
    p.add_argument("--lines", type=int, default=3000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    p.add_argument("--lexicon", default=str(ROOT / "models" / "lexicon.tsv.gz"))
    p.add_argument("--examples", type=int, default=12)
    a = p.parse_args()

    r = Restorer(a.model, device=a.device, lexicon=a.lexicon)
    lex = r.lexicon
    total_cats, examples = Counter(), defaultdict(Counter)
    confusions, freq_err, freq_all, where = Counter(), Counter(), Counter(), Counter()

    print(f"{'set':>13} {'word%':>7} {'sent%':>7} {'errors':>7}  " + " ".join(f"{c:>11}" for c in CATS))
    for s in SETS:
        targets = short_set(a.lines) if s == "short" else load_eval_set(s, a.lines)
        outputs = r.restore_many([deaccent(t) for t in targets])
        res = score(targets, outputs)
        cats = Counter()
        for t, o in zip(targets, outputs):
            for m in WORD_RE.finditer(t):
                gw, ow = m.group(), o[m.start() : m.end()]
                zb = min(7, int(_zipf(gw.lower(), "hu")))
                freq_all[zb] += 1
                if gw == ow:
                    continue
                freq_err[zb] += 1
                c = categorize(gw, ow, m.start(), t, lex)
                cats[c] += 1
                examples[c][f"{gw}->{ow}"] += 1
                diffs = [i for i, (x, y) in enumerate(zip(gw, ow)) if x != y]
                for i in diffs:
                    confusions[(gw[i].lower(), ow[i].lower())] += 1
                where["stem" if diffs[0] < max(3, len(gw) * 0.6) else "suffix"] += 1
        total_cats.update(cats)
        n = sum(cats.values())
        print(f"{s:>13} {res['word_acc'] * 100:7.2f} {res['sentence_acc'] * 100:7.2f} {n:7d}  "
              + " ".join(f"{cats[c] / max(n, 1) * 100:10.1f}%" for c in CATS), flush=True)

    n = sum(total_cats.values())
    print(f"\nall errors: {n}")
    for c in CATS:
        top = ", ".join(f"{e} ({k})" if k > 1 else e for e, k in examples[c].most_common(a.examples))
        print(f"  {c:>11} {total_cats[c]:5d} {total_cats[c] / n * 100:5.1f}%  {top}")
    print("\nvowel confusions (gold -> output):",
          ", ".join(f"{x}->{y} {k}" for (x, y), k in confusions.most_common(14)))
    print("error position:", ", ".join(f"{k} {v / sum(where.values()) * 100:.0f}%" for k, v in where.items()))
    print("word error rate by Hungarian zipf frequency (0 = unknown, 7 = very common):")
    for z in sorted(freq_all):
        print(f"  zipf {z}: {freq_err[z] / freq_all[z] * 100:5.2f}% of {freq_all[z]} words")


if __name__ == "__main__":
    main()
