"""Drop corpus lines that are probably missing accents -> data/clean/*.u16

Web text often has words typed without accents ("amig", "szeles"), which teaches the model
to leave accents off. A lowercase word counts as suspicious when the lexicon has an accented
form of it that is far more common (e.g. "amíg" 50x more often than "amig"). A line is dropped
if it has one strongly suspicious word or two weakly suspicious ones. Capitalized words (names)
and common English words are never counted. Test/val files are cleaned the same way so the
"clean" evaluation measures the model, not the reference's typos.

    uv run python scripts/clean_data.py [--procs 24]
"""

from __future__ import annotations

import argparse
import glob
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.lexicon import TOKEN_RE, Lexicon  # noqa: E402

SRC, DST = ROOT / "data", ROOT / "data" / "clean"
CHUNK = 20_000_000  # codepoints per task
WEAK, STRONG = 1, 2
SUS: dict[str, int] = {}


# accent "strength" per vowel; a form is under-accented if every differing vowel is weaker
_RANK = {c: i for chain in ("aá", "eé", "ií", "oóő", "oöő", "uúű", "uüű") for i, c in enumerate(chain)}
_INCOMPARABLE = {frozenset("óö"), frozenset("úü")}


def under_accented(form: str, ref: str) -> bool:
    diff = [(x, y) for x, y in zip(form, ref) if x != y]
    return bool(diff) and all(frozenset((x, y)) not in _INCOMPARABLE and _RANK[x] < _RANK[y] for x, y in diff)


def suspicious_words(lex: Lexicon, weak_ratio=20, strong_ratio=50) -> dict[str, int]:
    """Forms like "amig"/"kiállitásra"/"megerösödtek": a more-accented spelling is far more common."""
    from wordfreq import zipf_frequency

    out = {}
    for key, forms in lex.entries.items():
        top, top_c = forms[0]  # sorted by count
        for w, c in forms[1:]:
            # c >= 1000: common enough to be a real word form ("hogy tudjak" vs "tudják")
            if top_c < 50 or top_c < weak_ratio * c or c >= 1000 or not under_accented(w, top):
                continue
            if w == key and zipf_frequency(key, "en") >= 3.5:  # "video", "film": English usage is legitimate
                continue
            out[w] = STRONG if top_c >= 100 and top_c >= strong_ratio * c else WEAK
    return out


def init(sus):
    global SUS
    SUS = sus


def clean_chunk(args):
    path, start, end = args
    a = np.memmap(path, dtype=np.uint16, mode="r")
    text = a[start:end].tobytes().decode("utf-16-le", "replace")
    kept, dropped = [], 0
    for line in text.split("\n"):
        score = 0
        for m in TOKEN_RE.finditer(line):
            w = m.group()
            if w.islower():
                score += SUS.get(w, 0)
        if score >= 2:
            dropped += 1
        else:
            kept.append(line)
    return "\n".join(kept).encode("utf-16-le"), dropped, len(kept) + dropped


def tasks(path: str):
    a = np.memmap(path, dtype=np.uint16, mode="r")
    bounds, pos = [0], 0
    while pos + CHUNK < len(a):
        nl = np.flatnonzero(a[pos + CHUNK : pos + CHUNK + 100_000] == 10)
        pos = pos + CHUNK + (int(nl[0]) if len(nl) else 0)
        bounds.append(pos + 1)
    bounds.append(len(a))
    return [(path, s, e) for s, e in zip(bounds, bounds[1:])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--only", default="*", help="glob of files in data/ to clean")
    p.add_argument("--lexicon", default=str(ROOT / "models" / "lexicon.tsv.gz"))
    a = p.parse_args()
    sus = suspicious_words(Lexicon.load(a.lexicon))
    print(f"suspicious words: {len(sus)} ({sum(v == STRONG for v in sus.values())} strong)", flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs, initializer=init, initargs=(sus,)) as pool:
        for path in sorted(glob.glob(str(SRC / f"{a.only}.u16"))):
            dropped = total = 0
            with open(DST / Path(path).name, "wb") as f:
                for data, d, t in pool.imap(clean_chunk, tasks(path)):
                    f.write(data)  # chunks end with their own newline
                    dropped, total = dropped + d, total + t
            print(f"{Path(path).name:32s} dropped {dropped:>10,} / {total:>11,} lines ({dropped / max(total, 1) * 100:.1f}%)",
                  flush=True)


if __name__ == "__main__":
    main()
