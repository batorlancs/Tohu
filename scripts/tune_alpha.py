"""Evaluate model + lexicon combination for several alpha values (alpha=0 -> model only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.chars import deaccent, normalize  # noqa: E402
from tohu.evaluate import load_eval_set, score  # noqa: E402
from tohu.restore import Restorer  # noqa: E402

SETS = ("web-formal", "web-informal", "subtitles", "wikipedia", "challenge", "hunglish")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "tohu.pt"))
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.5")
    p.add_argument("--lines", type=int, default=1500)
    p.add_argument("--errors", type=float, help="print top errors for this alpha")
    p.add_argument("--min-totals", default="10", help="comma list of lexicon min counts to try")
    a = p.parse_args()
    r = Restorer(a.model, device="mps")
    data = {}
    for s in SETS:
        t = load_eval_set(s, a.lines)
        inp = [normalize(deaccent(x)) for x in t]
        data[s] = (t, inp, r._logits(inp))
    print(f"{'alpha':>6} {'min':>4} " + " ".join(f"{s:>13}" for s in SETS) + "    avg")
    grid = [(al, mt) for mt in map(int, a.min_totals.split(",")) for al in map(float, a.alphas.split(","))]
    for alpha, mt in grid:
        r.alpha, r.min_total = alpha, mt
        accs = []
        for s in SETS:
            t, inp, lg = data[s]
            res = score(t, [r._decode(x, l)[0] for x, l in zip(inp, lg)])
            accs.append(res["word_acc"] * 100)
            if a.errors is not None and alpha == a.errors:
                print(s, [f"{g}->{o}" for (g, o), _ in res["errors"].most_common(40)])
        print(f"{alpha:>6} {mt:>4} " + " ".join(f"{x:>13.2f}" for x in accs) + f"  {sum(accs) / len(accs):.2f}", flush=True)


if __name__ == "__main__":
    main()
