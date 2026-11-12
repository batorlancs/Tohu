"""Export everything the macOS app needs into models/export/, then check it against PyTorch.

    uv run --group export python scripts/export_coreml.py [--model models/tohu.pt]

Writes:
  Tohu.mlpackage   Core ML model: ids (1, T) int32 -> logits (1, T, 4), T in BUCKETS
  Tohu.mlmodelc    the same, compiled (what the app loads; no Xcode needed at build time)
  lexicon.bin      sorted, memory-mappable lexicon (format below)
  guard.json       wordfreq numbers the English guard needs
  manifest.json    alphabet, vowel variants and decoding constants
  golden.jsonl     input -> expected output pairs, used by the Swift tests

lexicon.bin layout (little-endian):
  "TOHULEX1" | u32 count | u32 offsets[count + 1] | blob
  blob[offsets[i]:offsets[i+1]] = UTF-8 "key\\tform:count form:count ...", sorted by key bytes
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tohu.chars import BASE_VOWELS, CHARS, NUM_CLASSES, VARIANTS, deaccent, encode_input, normalize  # noqa: E402
from tohu.lexicon import DEFAULT_LEXICON, Lexicon  # noqa: E402
from tohu.model import AccentModel  # noqa: E402
from tohu.restore import DEFAULT_MODEL, WORD_RE, Restorer  # noqa: E402

BUCKETS = [32, 64, 128, 256, 512]
CHUNK, CTX = 384, 64  # Restorer._logits windowing; CHUNK + 2 * CTX must fit the largest bucket
assert CHUNK + 2 * CTX <= BUCKETS[-1]


# ---------------------------------------------------------------- model
def _patch_coremltools() -> None:
    """coremltools 9.0 casts constant (1,)-shaped tensors with int(x) / bool(x), which NumPy 2.5
    rejects. Same logic as coremltools' _cast, but taking .item() of single-element constants."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if not (len(x.shape) == 0 or all(d == 1 for d in x.shape)):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            res = mb.const(val=dtype(np.asarray(x.val).item()), name=node.name)
        else:
            if len(x.shape) > 0:
                x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    ops._cast = _cast


def export_model(model: AccentModel, out: Path, precision: str) -> None:
    import coremltools as ct

    _patch_coremltools()
    example = torch.zeros(1, BUCKETS[-1], dtype=torch.int32)
    traced = torch.jit.trace(model.eval(), example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ids", shape=ct.EnumeratedShapes([(1, b) for b in BUCKETS]), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    ml.short_description = "tohu: Hungarian accent restoration (per-character vowel variant logits)"
    if out.exists():
        shutil.rmtree(out)
    ml.save(str(out))
    compiled = out.with_suffix(".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    ct.utils.compile_model(str(out), str(compiled))


class CoreMLRestorer(Restorer):
    """Restorer whose network runs in Core ML, one window at a time, padded to a bucket.
    This is the exact procedure the Swift side reproduces."""

    def __init__(self, mlmodel, **kw):
        super().__init__(**kw)
        self.ml = mlmodel

    def _logits(self, texts, chunk=CHUNK, ctx=CTX):
        out = []
        for t in texts:
            lg = np.zeros((len(t), NUM_CLASSES), dtype=np.float32)
            pos = 0
            while pos < len(t):
                end = min(len(t), pos + chunk)
                if end < len(t):
                    sp = t.rfind(" ", pos + chunk // 2, end)
                    end = sp if sp > 0 else end
                ws, we = max(0, pos - ctx), min(len(t), end + ctx)
                ids = encode_input(t[ws:we])
                bucket = next(b for b in BUCKETS if b >= len(ids))
                padded = np.zeros((1, bucket), dtype=np.int32)
                padded[0, : len(ids)] = ids
                y = self.ml.predict({"ids": padded})["logits"][0]
                lg[pos:end] = y[pos - ws : end - ws]
                pos = end
            out.append(lg)
        return out


# ---------------------------------------------------------------- lexicon
def export_lexicon(src: Path, out: Path) -> int:
    lines = []
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                lines.append(line.encode("utf-8"))
    lines.sort(key=lambda b: b.split(b"\t", 1)[0])
    offsets = np.zeros(len(lines) + 1, dtype="<u4")
    np.cumsum([len(b) for b in lines], out=offsets[1:])
    with open(out, "wb") as f:
        f.write(b"TOHULEX1")
        f.write(struct.pack("<I", len(lines)))
        f.write(offsets.tobytes())
        for b in lines:
            f.write(b)
    return len(lines)


# ---------------------------------------------------------------- English guard
def export_guard(out: Path) -> tuple[int, int]:
    """english_guard() fires when zipf_en(w) >= 3 and zipf_hu(accented w) < 1.5 and zipf_hu(w) < zipf_en(w).
    So we need: every letter-word with zipf_en >= 3 (with its en and hu zipf), and every Hungarian
    word with zipf_hu >= 1.5 that is an accent variant of one of them."""
    from wordfreq import get_frequency_dict, zipf_frequency

    en = {}
    for w, freq in get_frequency_dict("en").items():
        # zipf is rounded to 2 decimals, so look slightly below the cut-off and let zipf decide
        if freq < 10 ** (2.99 - 9) or not WORD_RE.fullmatch(w) or w != w.lower():
            continue
        z = zipf_frequency(w, "en")
        if z >= 3.0:
            en[w] = [z, zipf_frequency(w, "hu")]
    keys = {deaccent(w) for w in en}
    hu_known = sorted(
        w for w, freq in get_frequency_dict("hu").items()
        if freq >= 10 ** (1.49 - 9) and deaccent(w) in keys and zipf_frequency(w, "hu") >= 1.5
    )
    out.write_text(json.dumps({"en": en, "hu_known": hu_known}, ensure_ascii=False, separators=(",", ":")))
    return len(en), len(hu_known)


# ---------------------------------------------------------------- checks
def challenge_lines() -> list[str]:
    lines = (ROOT / "tests" / "challenge.txt").read_text(encoding="utf-8").splitlines()
    return [l for l in lines if l.strip() and not l.startswith("#")]


def word_acc(pred: str, gold: str) -> float:
    pw, gw = WORD_RE.findall(pred), WORD_RE.findall(gold)
    return sum(p == g for p, g in zip(pw, gw)) / max(1, len(gw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--lexicon", default=str(DEFAULT_LEXICON))
    ap.add_argument("--out", default=str(ROOT / "models" / "export"))
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    a = ap.parse_args()

    import coremltools as ct

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = AccentModel.load(a.model)
    # _init() re-randomises the padding_idx row, so padded inputs leak a random vector into the
    # conv stem and from there into every position. Pads are only ever id 0, so zeroing that row
    # makes a padded window give exactly the same logits as the unpadded text.
    with torch.no_grad():
        model.emb.weight[0].zero_()

    t = time.perf_counter()
    export_model(model, out / "Tohu.mlpackage", a.precision)
    print(f"model     -> {out / 'Tohu.mlpackage'}  ({time.perf_counter() - t:.1f}s, {a.precision})")
    n = export_lexicon(Path(a.lexicon), out / "lexicon.bin")
    print(f"lexicon   -> {out / 'lexicon.bin'}  ({n:,} keys, {(out / 'lexicon.bin').stat().st_size / 1e6:.1f} MB)")
    ne, nh = export_guard(out / "guard.json")
    print(f"guard     -> {out / 'guard.json'}  ({ne:,} English words, {nh:,} Hungarian look-alikes)")

    ref = Restorer(model)
    manifest = {
        "version": 1,
        "chars": CHARS,  # id = index + 2; 0 = pad, 1 = unknown
        "variants": VARIANTS,
        "normalize": {"õ": "ő", "Õ": "Ő", "û": "ű", "Û": "Ű", "ô": "ő", "Ô": "Ő"},
        "buckets": BUCKETS,
        "chunk": CHUNK,
        "ctx": CTX,
        "alpha": ref.alpha,
        "min_total": ref.min_total,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    # --- parity: Core ML vs PyTorch on the challenge set (and one long text to exercise windowing)
    ml = ct.models.MLModel(str(out / "Tohu.mlpackage"), compute_units=ct.ComputeUnit.ALL)
    cm = CoreMLRestorer(ml, model=model)
    gold = challenge_lines()
    inputs = [deaccent(g) for g in gold] + [" ".join(deaccent(g) for g in gold)]

    max_diff = 0.0
    for b in BUCKETS:
        ids = np.random.default_rng(b).integers(2, len(CHARS) + 2, size=(1, b)).astype(np.int32)
        y_ml = ml.predict({"ids": ids})["logits"]
        with torch.inference_mode():
            y_pt = model(torch.from_numpy(ids).long()).numpy()
        max_diff = max(max_diff, float(np.abs(y_ml - y_pt).max()))
    print(f"\nmax |logit diff| Core ML vs PyTorch over buckets {BUCKETS}: {max_diff:.2e}")

    t = time.perf_counter()
    pt_out = [ref.restore(x) for x in inputs]
    t_pt = time.perf_counter() - t
    t = time.perf_counter()
    ml_out = [cm.restore(x) for x in inputs]
    t_ml = time.perf_counter() - t
    mismatches = [(i, p, m) for i, (p, m) in enumerate(zip(pt_out, ml_out)) if p != m]
    acc_pt = np.mean([word_acc(p, g) for p, g in zip(pt_out, gold)])
    acc_ml = np.mean([word_acc(m, g) for m, g in zip(ml_out, gold)])
    print(f"challenge set: {len(gold)} lines + 1 long text ({len(inputs[-1])} chars)")
    print(f"  word accuracy  PyTorch {acc_pt:.2%}   Core ML {acc_ml:.2%}")
    print(f"  identical outputs: {len(inputs) - len(mismatches)}/{len(inputs)}")
    print(f"  time  PyTorch {t_pt * 1000:.0f} ms   Core ML {t_ml * 1000:.0f} ms  (incl. decoding)")
    for i, p, m in mismatches[:10]:
        print(f"    pt: {p[:100]}\n    ml: {m[:100]}")

    # --- golden file for the Swift tests (expected outputs come from the Core ML path)
    extra = ["Kerlek KULDD at a deadline-ra a reportot!", "2010-tol 2020-ig", "", "  szia\n\nhogy vagy?  ",
             "Az o haza ott van, a hazbol jott ki.", "Ékezetes marad: kérlek kuldd"]
    with open(out / "golden.jsonl", "w", encoding="utf-8") as f:
        for x in inputs + extra:
            y, words = cm.analyze(x)
            f.write(json.dumps({"input": x, "output": y,
                                "words": [{"start": w.start, "end": w.end, "text": w.text,
                                           "confidence": round(w.confidence, 4)} for w in words]},
                               ensure_ascii=False) + "\n")
    print(f"\ngolden    -> {out / 'golden.jsonl'}  ({len(inputs) + len(extra)} cases)")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
