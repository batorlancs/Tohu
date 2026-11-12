"""Fine-tune the model on your own correctly accented text (e.g. corrections collected by the app).

  uv run python scripts/finetune.py my_texts.txt [--steps 600]

Your text is mixed with the original corpora (replay) so the model adapts to your
style and vocabulary without forgetting general Hungarian. Writes models/tohu_ft.pt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.chars import normalize  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("texts", help="UTF-8 file of correctly accented text")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--weight", type=float, default=0.3, help="share of batches drawn from your text")
    p.add_argument("--init", default=str(ROOT / "models" / "tohu.pt"))
    a = p.parse_args()

    text = normalize(Path(a.texts).read_text(encoding="utf-8")).strip()
    paras = [ln.strip() for ln in text.split("\n") if ln.strip()]
    body = "\n\n".join(paras) + "\n\n"
    cp = np.frombuffer(body.encode("utf-32-le"), dtype=np.uint32)
    np.where(cp < 0x10000, cp, 0xFFFD).astype(np.uint16).tofile(ROOT / "data" / "user_finetune.u16")
    print(f"{len(paras)} paragraphs, {len(body)} chars of your text")
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", "tohu_ft", "--init", a.init,
           "--steps", str(a.steps), "--lr", "2e-4", "--warmup", "50", "--eval-every", str(a.steps),
           "--extra", "user_finetune.u16", "--extra-weight", str(a.weight)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
