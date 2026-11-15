"""Use a trained model to drop training lines with accent typos -> data/clean2/*.u16

The lexicon filter (clean_data.py) misses typos that are themselves common ("elött", "öket",
"legyözni": ö typed for ő) and rare words. A trained model spots them: a word is a typo if the
model gives its written form < 0.1% probability, the model's own reading is far more likely, and
the written form has *fewer* accents than the model's reading (typos drop accents; the reverse
is usually a name). Common English words are never flagged (Hunglish lines stay).

Only training files are filtered. Test/val files are linked unchanged from the source folder,
so evaluation is not biased towards the model used for cleaning.

    TOHU_DATA=data/clean python scripts/model_clean.py --model models/tohu.pt [--out data/clean2]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.chars import _BASE_TABLE, _CHAR_TO_ID, _LABEL_TABLE, IGNORE, VARIANTS  # noqa: E402
from tohu.data import DATA  # noqa: E402
from tohu.model import AccentModel  # noqa: E402

BUCKETS = (64, 128, 256, 512)
TOKENS = 262144  # characters per batch
LOG_P = float(np.log(1e-3))
MARGIN = 6.0  # nats between the model's best reading and the written form

# number of valid classes per codepoint (0 = not a vowel), letter table
_NVALID = np.zeros(0x10000, dtype=np.int64)
for _b, _v in VARIANTS.items():
    for _c in _v:
        _NVALID[ord(_c)] = _NVALID[ord(_c.upper())] = len(_v)
_LETTER = np.array([chr(i).isalpha() for i in range(0x10000)], dtype=bool)


def under(gold: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """gold class is a weaker accent than pred (o<ó<ő, o<ö<ő, a<á ...); ó vs ö are incomparable."""
    return ((gold == 0) & (pred > 0)) | (((gold == 1) | (gold == 2)) & (pred == 3))


@torch.inference_mode()
def flag_segments(model, a: np.ndarray, segs: np.ndarray, L: int, dev) -> list[tuple[int, str]]:
    """segs: (n, 2) start/end into a, all of length <= L. Returns [(segment row, flagged word)]."""
    n = len(segs)
    lens = segs[:, 1] - segs[:, 0]
    idx = segs[:, 0:1] + np.arange(L)[None, :]
    valid = np.arange(L)[None, :] < lens[:, None]
    cp = np.where(valid, a[np.minimum(idx, len(a) - 1)], 0).astype(np.int64)
    ids = np.where(valid, _CHAR_TO_ID[_BASE_TABLE[cp]], 0)
    labels = np.where(valid, _LABEL_TABLE[cp], IGNORE)
    nval = np.where(valid, _NVALID[cp], 0)
    letter = valid & _LETTER[cp]

    ids_t = torch.from_numpy(ids).to(dev, non_blocking=True)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = model(ids_t).float()
    nval_t = torch.from_numpy(nval).to(dev)
    cls = torch.arange(4, device=dev)
    logits = logits.masked_fill(cls[None, None, :] >= nval_t[..., None].clamp(min=1), float("-inf"))
    lp = torch.log_softmax(logits, -1)
    lab = torch.from_numpy(labels).to(dev)
    is_v = (lab != IGNORE) & (nval_t > 0)
    lab0 = lab.clamp(min=0)
    gold = torch.where(is_v, lp.gather(-1, lab0[..., None])[..., 0], 0.0)
    best, pred = lp.max(-1)
    best = torch.where(is_v, best, 0.0)
    differ = is_v & (pred != lab0)
    bad_char = differ & ~under(lab0, pred)  # a differing char where the text has MORE accent

    # word ids: runs of letters
    let = torch.from_numpy(letter).to(dev)
    start = let & ~torch.nn.functional.pad(let[:, :-1], (1, 0), value=False)
    wid = torch.cumsum(start.long(), 1) * let  # 0 = not in a word
    W = int(wid.max().item()) + 1
    wg = torch.zeros(n, W, device=dev).scatter_add_(1, wid, gold)
    wb = torch.zeros(n, W, device=dev).scatter_add_(1, wid, best)
    wbad = torch.zeros(n, W, device=dev).scatter_add_(1, wid, bad_char.float())
    flag = (wg < LOG_P) & (wb - wg > MARGIN) & (wbad == 0)
    flag[:, 0] = False
    hits = flag.nonzero().cpu().numpy()
    if len(hits) == 0:
        return []
    # recover the flagged words' text (few) for the English check
    wid_np = wid.cpu().numpy()
    words = []
    for r, w in hits:
        pos = np.flatnonzero(wid_np[r] == w)
        words.append((r, "".join(map(chr, cp[r, pos]))))
    return words


@lru_cache(maxsize=None)
def english_ok(word: str) -> bool:
    from wordfreq import zipf_frequency

    return zipf_frequency(word.lower(), "en") >= 3.5


def clean_file(model, src: Path, dst: Path, dev) -> tuple[int, int, list[str]]:
    a = np.fromfile(src, dtype=np.uint16)
    nl = np.flatnonzero(a == 10)
    starts = np.concatenate([[0], nl + 1])
    ends = np.concatenate([nl, [len(a)]])
    n_lines = len(starts)
    # split lines into segments of <= 512 chars
    seg_list, owner = [], []
    for li in np.flatnonzero(ends - starts > 0):
        s, e = int(starts[li]), int(ends[li])
        while s < e:
            seg_list.append((s, min(e, s + BUCKETS[-1])))
            owner.append(li)
            s += BUCKETS[-1]
    segs = np.array(seg_list, dtype=np.int64).reshape(-1, 2)
    owner = np.array(owner, dtype=np.int64)
    lens = segs[:, 1] - segs[:, 0]
    drop = np.zeros(n_lines, dtype=bool)
    examples = []
    for i, L in enumerate(BUCKETS):
        lo = BUCKETS[i - 1] if i else 0
        sel = np.flatnonzero((lens > lo) & (lens <= L))
        B = TOKENS // L
        for b in range(0, len(sel), B):
            chunk = sel[b : b + B]
            batch = segs[chunk]
            if len(batch) < B:  # pad with empty segments: one shape per bucket for torch.compile
                batch = np.concatenate([batch, np.zeros((B - len(batch), 2), dtype=np.int64)])
            words = flag_segments(model, a, batch, L, dev)
            for r, w in words:
                if not english_ok(w):
                    drop[owner[chunk[r]]] = True
                    if len(examples) < 30:
                        examples.append(w)
    keep_char = np.repeat(~drop, ends - starts + 1)[: len(a)]  # each line + its newline
    a[keep_char].tofile(dst)
    return int(drop.sum()), n_lines, examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "tohu.pt"))
    p.add_argument("--out", default=str(ROOT / "data" / "clean2"))
    p.add_argument("--only", default="*")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AccentModel.load(a.model).to(dev).eval()
    with torch.no_grad():
        model.emb.weight[0].zero_()  # padding id 0 must not leak into the conv stem
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        model = torch.compile(model, dynamic=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted(glob.glob(str(DATA / f"{a.only}.u16"))):
        src = Path(src)
        dst = out / src.name
        if any(t in src.name for t in ("_test", "_val")):
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            continue
        if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
            print(f"{src.name:34s} already done", flush=True)
            continue
        t = time.time()
        d, n, ex = clean_file(model, src, dst, dev)
        mc = src.stat().st_size / 2 / 1e6
        print(f"{src.name:34s} dropped {d:>9,} / {n:>10,} lines ({d / max(n, 1) * 100:4.1f}%)  "
              f"{mc / (time.time() - t):5.1f}M chars/s  e.g. {' '.join(ex[:8])}", flush=True)


if __name__ == "__main__":
    main()
