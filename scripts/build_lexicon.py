"""Count accented word forms over all training corpora -> models/lexicon.tsv.gz

Reads from $TOHU_DATA (default data/). Usage: build_lexicon.py [--out PATH] [--procs N]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.chars import deaccent  # noqa: E402
from tohu.data import DATA  # noqa: E402
from tohu.lexicon import DEFAULT_LEXICON, TOKEN_RE  # noqa: E402

CHUNK = 50_000_000  # chars


def count_file(path: str) -> Counter:
    a = np.memmap(path, dtype=np.uint16, mode="r")
    total: Counter = Counter()
    for i in range(0, len(a), CHUNK):
        text = a[i : i + CHUNK].tobytes().decode("utf-16-le", "replace")
        total.update(m.group().lower() for m in TOKEN_RE.finditer(text))
        if len(total) > 6_000_000:  # keep memory bounded
            total = Counter({w: c for w, c in total.items() if c > 1})
    print(Path(path).name, len(total), flush=True)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_LEXICON))
    p.add_argument("--procs", type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    files = sorted(glob.glob(str(DATA / "fineweb_*_0*.u16"))) + [
        str(DATA / "subs_train.u16"),
        str(DATA / "wiki_train.u16"),
    ]
    counts: Counter = Counter()
    with Pool(a.procs) as pool:
        for c in pool.imap_unordered(count_file, files):
            counts.update(c)
    by_key: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for w, c in counts.items():
        if c >= 2:
            by_key[deaccent(w)].append((w, c))
    out.parent.mkdir(exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as f:
        for k, forms in by_key.items():
            if sum(c for _, c in forms) >= 3:
                forms.sort(key=lambda x: -x[1])
                f.write(k + "\t" + " ".join(f"{w}:{c}" for w, c in forms) + "\n")
    print("keys:", len(by_key), "size MB:", out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
