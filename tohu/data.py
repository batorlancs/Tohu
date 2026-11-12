"""Streams random training windows from the uint16 corpora, augmented on the fly."""

from __future__ import annotations

import glob
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .augment import augment
from .chars import make_example

DATA = Path(__file__).resolve().parents[1] / "data"

# source glob -> sampling weight
MIX = {
    "fineweb_formal_0*.u16": 0.38,
    "fineweb_informal_0*.u16": 0.27,
    "subs_train.u16": 0.22,
    "wiki_train.u16": 0.13,
}


def load_text(path: Path) -> str:
    return np.fromfile(path, dtype=np.uint16).tobytes().decode("utf-16-le")


class Corpus:
    def __init__(self, pattern: str):
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in sorted(glob.glob(str(DATA / pattern)))]
        if not self.arrays:
            raise FileNotFoundError(pattern)
        self.sizes = np.array([len(a) for a in self.arrays], dtype=np.float64)

    def window(self, rng: random.Random, length: int) -> str:
        a = self.arrays[rng.choices(range(len(self.arrays)), weights=self.sizes)[0]]
        i = rng.randrange(0, max(1, len(a) - length - 64))
        s = a[i : i + length + 64].tobytes().decode("utf-16-le", errors="replace")
        # snap to word boundaries; avoid spanning document breaks
        start = s.find(" ") + 1 if i > 0 else 0
        s = s[start : start + length]
        if "\n\n" in s:
            s = max(s.split("\n\n"), key=len)
        cut = s.rfind(" ")
        return s[:cut] if cut > length // 2 else s


# Length buckets (fixed shapes for torch.compile, little padding). Batch size keeps ~constant chars/step.
BUCKETS = {64: 0.15, 128: 0.25, 192: 0.25, 256: 0.20, 320: 0.15}


class TrainStream(IterableDataset):
    """Yields whole (ids, labels) batches of shape (tokens_per_batch // L, L) for a random bucket L."""

    def __init__(self, tokens_per_batch: int = 24576, seed: int = 0, augment_p: float = 1.0, mix: dict | None = None):
        self.tokens = tokens_per_batch
        self.seed = seed
        self.augment_p = augment_p
        self.mix = mix or MIX

    def sample(self, rng, corpora, L: int):
        c = corpora[rng.choices(list(self.mix), list(self.mix.values()))[0]]
        text = c.window(rng, rng.randint(max(8, L - 64), L - 8)).strip()
        if len(text) < 2:
            return None
        target, inp = augment(text, rng, short_p=0.4 if L == 64 else 0.0) if rng.random() < self.augment_p else (text, None)
        return make_example(target[:L], inp[:L] if inp is not None else None)

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        corpora = {k: Corpus(k) for k in self.mix}
        while True:
            L = rng.choices(list(BUCKETS), list(BUCKETS.values()))[0]
            B = self.tokens // L
            ids = np.zeros((B, L), dtype=np.int64)
            labels = np.full((B, L), -100, dtype=np.int64)
            i = 0
            while i < B:
                ex = self.sample(rng, corpora, L)
                if ex is None:
                    continue
                ids[i, : len(ex[0])], labels[i, : len(ex[1])] = ex
                i += 1
            yield torch.from_numpy(ids), torch.from_numpy(labels)
