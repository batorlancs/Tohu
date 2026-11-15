"""Train (or fine-tune with --init) the accent restoration model on CUDA, Apple Silicon (MPS) or CPU.

CUDA example (desk):  TOHU_DATA=data/clean python scripts/train.py --name big --dim 512 --layers 8 \
    --heads 8 --amp --tokens 65536 --steps 120000 --workers 20 --buckets 64:.1,128:.2,256:.3,384:.2,512:.2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tohu.data import BUCKETS, MIX, TrainStream  # noqa: E402
from tohu.evaluate import evaluate  # noqa: E402
from tohu.model import AccentModel, Config  # noqa: E402
from tohu.restore import Restorer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="tohu")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--tokens", type=int, default=24576, help="characters per batch")
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-lines", type=int, default=600)
    p.add_argument("--init", help="checkpoint to start from (fine-tuning)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--extra", help="extra corpus file in data/ (e.g. your own text) for fine-tuning")
    p.add_argument("--extra-weight", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--buckets", help="length:weight list, e.g. 64:.1,128:.2,256:.3,384:.2,512:.2")
    p.add_argument("--resume", action="store_true", help="continue from models/<name>_state.pt")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights, e.g. 0.9999 (0 = off); "
                   "evaluation and the saved checkpoints then use the averaged weights")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    state_path = out_dir / f"{a.name}_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False) if a.resume else None
    if a.init:
        model = AccentModel.load(a.init)
    else:
        model = AccentModel(Config(dim=a.dim, layers=a.layers, heads=a.heads, dropout=a.dropout))
    if state:
        model.load_state_dict(state["model"]["state"])
    model.to(dev).train()
    ema = None
    if a.ema:
        ema = AccentModel(model.cfg).to(dev).eval()
        ema.load_state_dict(state["ema"] if state and "ema" in state else model.state_dict())
        ema.requires_grad_(False)
        ema_p, model_p = list(ema.parameters()), list(model.parameters())
    print(f"params: {model.num_params() / 1e6:.2f}M  device: {dev}", flush=True)
    fmodel = model if a.no_compile else torch.compile(model, dynamic=False)

    mix = dict(MIX)
    if a.extra:
        mix = {k: v * (1 - a.extra_weight) for k, v in MIX.items()} | {a.extra: a.extra_weight}
    buckets = {int(k): float(v) for k, v in (x.split(":") for x in a.buckets.split(","))} if a.buckets else BUCKETS
    start = state["step"] + 1 if state else 1
    loader = DataLoader(
        TrainStream(tokens_per_batch=a.tokens, seed=a.seed + start, mix=mix, buckets=buckets),
        batch_size=None,
        num_workers=a.workers,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=dev.type == "cuda",
    )
    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr,
        betas=(0.9, 0.98),
        fused=dev.type == "cuda",
    )
    if state:
        opt.load_state_dict(state["opt"])

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

    best, t0, chars, loss_avg = (state["best"] if state else -1.0), time.time(), 0, None
    if state:
        print(f"resumed at step {start} (best {best:.2f})", flush=True)
    it = iter(loader)
    for step in range(start, a.steps + 1):
        ids, labels = next(it)
        ids, labels = ids.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.amp):
            logits = fmodel(ids)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_lerp_(ema_p, model_p, 1.0 - min(a.ema, (1 + step) / (10 + step)))
        chars += int((ids != 0).sum())
        lv = loss.item() if step % 50 == 0 else None
        if lv is not None:
            loss_avg = lv if loss_avg is None else 0.9 * loss_avg + 0.1 * lv
            el = time.time() - t0
            print(f"step {step} loss {loss_avg:.4f} lr {lr_at(step):.2e} chars/s {chars / el:,.0f} "
                  f"elapsed {el / 60:.1f}m", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            if ema is not None:
                raw = evaluate(Restorer(model, device=str(dev)), n_lines=a.eval_lines)
                raw_avg = sum(v["word_acc"] for v in raw.values()) / len(raw) * 100
                print(f"EVAL-RAW step {step} avg {raw_avg:.2f}", flush=True)
            final = ema if ema is not None else model
            res = evaluate(Restorer(final, device=str(dev)), n_lines=a.eval_lines)
            model.train()
            summary = {k: round(v["word_acc"] * 100, 2) for k, v in res.items()}
            score = sum(summary.values()) / len(summary)
            print(f"EVAL step {step} word_acc {json.dumps(summary)} avg {score:.2f}", flush=True)
            torch.save(final.checkpoint(), out_dir / f"{a.name}_last.pt")
            if score > best:
                best = score
                torch.save(final.checkpoint(), out_dir / f"{a.name}.pt")
                print(f"saved best -> models/{a.name}.pt", flush=True)
            st = {"model": model.checkpoint(), "opt": opt.state_dict(), "step": step, "best": best}
            if ema is not None:
                st["ema"] = ema.state_dict()
            torch.save(st, state_path)
    print("done", flush=True)
    import os

    os._exit(0)  # don't hang on persistent dataloader workers


if __name__ == "__main__":
    main()
