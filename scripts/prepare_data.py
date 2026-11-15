"""Download + clean training text into compact uint16 codepoint files under data/.

Sources:
  fineweb  FineWeb-2 hun_Latn (2024 web crawl), split into formal / informal (forums, Q&A, blogs)
  wiki     Hungarian Wikipedia
  subs     OpenSubtitles Hungarian (conversational)
  english  English subtitle lines (for Hunglish augmentation)

The critical step is filtering: lots of Hungarian online text is typed WITHOUT
accents, which would teach the model wrong answers. We drop any paragraph that
contains an unaccented spelling of a word that must have accents (es, igy, utan, ...).
"""

from __future__ import annotations

import argparse
import gzip
import io
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import xxhash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tohu.chars import ACCENTED_CHARS, BASE_VOWELS, normalize  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

# Unaccented spellings of common words that always carry accents and are not
# themselves Hungarian words. Seeing one => the writer skipped accents.
MARKERS = set(
    """es igy utan kozott tobb elso uj jo ev evben peldaul miert nelkul eppen tenyleg kerdes ugy
    mar meg- szep penz kesz ido elott itt- ot- ok- koszonom koszi  amugy egyebkent szoval
    tudnal lenne- nehany kulonben foleg vegul egesz fel- hogy- leszek- nagyobb- kicsit- ujra
    talan biztos- lehet- kell- gyerek- en te- o- ok- oket mindig- soha- mindenki- semmi- valami-
    hianyzik ertem erted tudod- latod nezd  szerinted- szerintem- tobbi utana elotte 
      kesobb korabban iden holnap- tegnap- ma- jovo multkor regen hetvegen
    hetfo kedd-  csutortok pentek vasarnap mukodik mukodni kerlek kerem kerdezni
    """.split()
)
MARKERS = {m for m in MARKERS if not m.endswith("-")}
assert not MARKERS & {"alatt", "mellett", "felett", "szerda", "mondd", "kosz"}
FOREIGN_OK = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")
INFORMAL_URL = re.compile(
    r"gyakorikerdesek|forum|prohardver|mobilarena|logout\.hu|hoxa|sg\.hu|reddit|blog|hup\.hu|"
    r"itcafe|gamepod|nlcafe|velvet|divany|femina|urbandictionary|komment|tumblr|wattpad|moly\.hu|"
    r"napiszar|facebook|twitter|gyerekszoba|babaszoba|nosalty|ittlakunk|kerdes|chat|irc|lol\.hu|"
    r"comment|vélemény|velemeny|hozzaszol|topic|thread",
    re.I,
)
TOKEN_STRIP = ".,;:!?()[]{}\"'„”“’‘…*-–—/"


def line_ok(line: str) -> bool:
    if len(line) < 15:
        return len(line) > 0 and not any(t.strip(TOKEN_STRIP).lower() in MARKERS for t in line.split())
    toks = [t.strip(TOKEN_STRIP).lower() for t in line.split()]
    if any(t in MARKERS for t in toks):
        return False
    letters = vowels = accented = foreign = 0
    for c in line:
        if c.isalpha():
            letters += 1
            if c in ACCENTED_CHARS:
                accented += 1
                vowels += 1
            elif c in BASE_VOWELS:
                vowels += 1
            elif ord(c) > 127 and c not in FOREIGN_OK:
                foreign += 1
    if letters == 0:
        return False
    if foreign > max(1, letters * 0.01):  # another language (Slovak, Romanian, ...)
        return False
    if letters >= 80 and accented < 0.10 * vowels:  # long line with suspiciously few accents
        return False
    return True


class Writer:
    """Appends cleaned docs as uint16 codepoints, docs separated by a blank line."""

    def __init__(self, name: str, limit_chars: int):
        self.path = DATA / f"{name}.u16"
        self.f = open(self.path, "wb")
        self.limit = limit_chars
        self.n = 0

    @property
    def full(self) -> bool:
        return self.n >= self.limit

    def write(self, text: str):
        if self.full or not text:
            return
        text = text + "\n\n"
        a = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        a = np.where(a < 0x10000, a, 0xFFFD).astype(np.uint16)
        self.f.write(a.tobytes())
        self.n += len(a)

    def close(self):
        self.f.close()
        print(f"{self.path.name}: {self.n / 1e6:.1f}M chars", flush=True)


SEEN: set[int] = set()


def clean_doc(text: str, min_line: int = 0) -> str | None:
    lines = [ln.strip() for ln in normalize(text).split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    kept, bad = [], 0
    for ln in lines:
        if len(ln) < min_line:
            continue
        if not line_ok(ln):
            bad += 1
            continue
        if len(ln) > 40:  # drop boilerplate repeated across the web
            h = xxhash.xxh64_intdigest(ln.encode())
            if h in SEEN:
                continue
            SEEN.add(h)
        kept.append(ln)
    if bad > 0.2 * len(lines) or not kept:  # writer mostly ignores accents -> distrust all
        return None
    return "\n".join(kept)


# ------------------------------------------------------------------ sources
def fineweb(shard: str, split: str, formal_limit: int, informal_limit: int, formal_keep: float):
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/HuggingFaceFW/fineweb-2/data/hun_Latn/{split}/{shard}.parquet",
        split="train",
        streaming=True,
    )
    tag = shard if split == "train" else "test"
    wf = Writer(f"fineweb_formal_{tag}", formal_limit)
    wi = Writer(f"fineweb_informal_{tag}", informal_limit)
    rng = random.Random(shard)
    seen_docs = 0
    for ex in ds:
        seen_docs += 1
        informal = bool(INFORMAL_URL.search(ex["url"]))
        w = wi if informal else wf
        if w.full or (not informal and rng.random() > formal_keep):
            if wf.full and wi.full:
                break
            continue
        doc = clean_doc(ex["text"])
        if doc:
            w.write(doc)
        if seen_docs % 50000 == 0:
            print(f"[{tag}] docs={seen_docs} formal={wf.n / 1e6:.0f}M informal={wi.n / 1e6:.0f}M", flush=True)
    wf.close()
    wi.close()


def wiki(limit: int, val_limit: int, part: int = 3):
    """part 3 is the original file (train + every 100th article as val); parts 0-2 are extra train data."""
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/wikimedia/wikipedia/20231101.hu/train-{part:05d}-of-00004.parquet",
        split="train",
        streaming=True,
    )
    if part == 3:
        w, v = Writer("wiki_train", limit), Writer("wiki_val", val_limit)
    else:
        w, v = Writer(f"wiki_train_{part}", limit), None
    for i, ex in enumerate(ds):
        doc = clean_doc(ex["text"], min_line=30)
        if doc:
            (v if v is not None and i % 100 == 0 else w).write(doc)
        if w.full and (v is None or v.full):
            break
    w.close()
    if v is not None:
        v.close()


def _stream_gz_lines(url: str, max_bytes: int):
    """Lines from the first max_bytes of a remote .gz (a truncated stream is fine)."""
    proc = subprocess.Popen(["curl", "-s", "-r", f"0-{max_bytes}", url], stdout=subprocess.PIPE)
    gz = gzip.GzipFile(fileobj=proc.stdout)
    reader = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    try:
        yield from reader
    except (EOFError, OSError):
        pass
    finally:
        proc.kill()


def subs(limit: int, val_limit: int, max_bytes: int, part: int = 1):
    """part 1: the original train/val files. part 2: everything after where part 1 stopped
    (part 1 is replayed without saving, so the val documents never end up in part 2)."""
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/hu.txt.gz"
    if part == 1:
        w, v = Writer("subs_train", limit), Writer("subs_val", val_limit)
    else:
        w, v = _NullWriter(250_000_000), _NullWriter(3_000_000)  # replay part 1's limits
    buf, groups = [], 0
    for line in _stream_gz_lines(url, max_bytes):
        line = line.strip()
        if line:
            buf.append(line)
        if len(buf) == 12:  # consecutive lines of one movie form a "document"
            groups += 1
            doc = clean_doc("\n".join(buf))
            if doc:
                (v if groups % 100 == 0 else w).write(doc)
            buf = []
            if w.full and v.full:
                if part == 1 or isinstance(w, Writer):
                    break
                w, v = Writer("subs_train_2", limit), _NullWriter(1 << 62)  # part 1 replayed; now save
    w.close()
    v.close()


class _NullWriter:
    def __init__(self, limit: int):
        self.n, self.limit = 0, limit

    @property
    def full(self) -> bool:
        return self.n >= self.limit

    def write(self, text: str):
        if not self.full and text:
            self.n += len(text) + 2

    def close(self):
        pass


def english(n_lines: int):
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz"
    out, seen = [], set()
    ok = re.compile(r"^[A-Za-z][A-Za-z ,.'!?-]+$")
    for line in _stream_gz_lines(url, 60_000_000):
        line = line.strip()
        if ok.match(line) and 2 <= len(line.split()) <= 12 and line not in seen:
            seen.add(line)
            out.append(line)
            if len(out) >= n_lines:
                break
    (DATA / "english_phrases.txt").write_text("\n".join(out))
    print(f"english_phrases.txt: {len(out)} lines")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["fineweb", "fineweb-test", "wiki", "subs", "english"])
    p.add_argument("--shard", default="000_00000")
    p.add_argument("--part", type=int, help="wiki: parquet file 0-3 (default 3); subs: 1 or 2 (default 1)")
    a = p.parse_args()
    M = 1_000_000
    if a.source == "fineweb":
        fineweb(a.shard, "train", formal_limit=220 * M, informal_limit=80 * M, formal_keep=0.5)
    elif a.source == "fineweb-test":
        fineweb("000_00000", "test", formal_limit=4 * M, informal_limit=4 * M, formal_keep=1.0)
    elif a.source == "wiki":
        if a.part is None or a.part == 3:
            wiki(limit=120 * M, val_limit=3 * M)
        else:
            wiki(limit=1000 * M, val_limit=0, part=a.part)
    elif a.source == "subs":
        if a.part == 2:
            subs(limit=1000 * M, val_limit=0, max_bytes=2_000_000_000, part=2)
        else:
            subs(limit=250 * M, val_limit=3 * M, max_bytes=400_000_000)
    elif a.source == "english":
        english(300_000)
