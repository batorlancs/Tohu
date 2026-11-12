"""CLI:  uv run python -m tohu "szia, hogy vagy?"     or     echo "..." | uv run python -m tohu

  --explain     list uncertain words with alternatives
  --model PATH  use another checkpoint
"""

from __future__ import annotations

import argparse
import sys
import time

from .restore import DEFAULT_MODEL, Restorer


def main():
    p = argparse.ArgumentParser(prog="tohu")
    p.add_argument("text", nargs="*")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--explain", action="store_true")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--no-guard", action="store_true")
    a = p.parse_args()

    text = " ".join(a.text) if a.text else sys.stdin.read()
    r = Restorer(a.model, guard=not a.no_guard)
    t = time.perf_counter()
    out, words = r.analyze(text)
    ms = (time.perf_counter() - t) * 1000
    print(out)
    if a.explain:
        unsure = [w for w in words if w.confidence < a.threshold]
        print(f"\n[{len(text)} chars in {ms:.0f} ms, {len(unsure)} uncertain word(s)]", file=sys.stderr)
        for w in unsure:
            alts = ", ".join(f"{s} ({p:.0%})" for s, p in w.alternatives)
            print(f"  {w.text:<20} {w.confidence:.0%}  alt: {alts}", file=sys.stderr)


if __name__ == "__main__":
    main()
