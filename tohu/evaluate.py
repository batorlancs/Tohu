"""Accuracy metrics (same definitions as the literature) on held-out sets."""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from .chars import BASE_VOWELS, deaccent, normalize
from .data import load_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

EVAL_SETS = {
    "web-formal": "fineweb_formal_test.u16",
    "web-informal": "fineweb_informal_test.u16",
    "subtitles": "subs_val.u16",
    "wikipedia": "wiki_val.u16",
}


def load_eval_set(name: str, n_lines: int = 1500, seed: int = 0) -> list[str]:
    if name == "hunglish":
        # informal web + subtitle lines with English words (+ Hungarian suffixes) swapped in
        from .augment import hunglish

        rng = random.Random(seed + 7)
        base = load_eval_set("web-informal", n_lines // 2, seed) + load_eval_set("subtitles", n_lines // 2, seed)
        return [hunglish(t, rng) for t in base]
    if name == "challenge":
        lines = (ROOT / "tests" / "challenge.txt").read_text().split("\n")
        return [normalize(ln.strip()) for ln in lines if ln.strip() and not ln.startswith("#")]
    lines = [ln for ln in load_text(DATA / EVAL_SETS[name]).split("\n") if len(ln) >= 20]
    random.Random(seed).shuffle(lines)
    return lines[:n_lines]


def score(targets: list[str], outputs: list[str]) -> dict:
    vow = vow_ok = words = words_ok = sent_ok = 0
    errors = Counter()
    for t, o in zip(targets, outputs):
        assert len(t) == len(o), (t, o)
        for a, b in zip(t, o):
            if deaccent(a) in BASE_VOWELS:
                vow += 1
                vow_ok += a == b
        all_ok = True
        for m in WORD_RE.finditer(t):
            words += 1
            ow = o[m.start() : m.end()]
            if ow == m.group():
                words_ok += 1
            else:
                all_ok = False
                errors[(m.group(), ow)] += 1
        sent_ok += all_ok
    return {
        "vowel_acc": vow_ok / max(1, vow),
        "word_acc": words_ok / max(1, words),
        "sentence_acc": sent_ok / max(1, len(targets)),
        "words": words,
        "errors": errors,
    }


def evaluate(restorer, sets=("web-formal", "web-informal", "subtitles", "wikipedia", "challenge"), n_lines=1500):
    results = {}
    for name in sets:
        targets = load_eval_set(name, n_lines)
        outputs = restorer.restore_many([deaccent(t) for t in targets])
        results[name] = score(targets, outputs)
    return results


def baseline_copy(sets, n_lines=1500):
    """Accuracy if we change nothing (lower bound)."""
    return {s: score(t := load_eval_set(s, n_lines), [deaccent(x) for x in t]) for s in sets}
