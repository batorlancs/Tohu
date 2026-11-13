"""Inference: unaccented (or partially accented) text in, accented text out.

Pipeline per text:
  1. character model -> log-probabilities of each vowel variant
  2. per word: combine with the corpus lexicon prior (if the word is known)
       score(form) = model_logprob(form) + alpha * log P_lex(form | unaccented word)
  3. English guard: never accent a plainly English word that isn't Hungarian
Accents the user typed are always kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .chars import BASE_VOWELS, LABEL_OF, VALID_MASK, apply_label, encode_input, normalize
from .lexicon import DEFAULT_LEXICON, TOKEN_RE, Lexicon
from .model import AccentModel

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "tohu.pt"
WORD_RE = re.compile(r"[^\W\d_]+")


@dataclass
class Word:
    start: int
    end: int
    text: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)


@lru_cache(maxsize=50000)
def _zipf(word: str, lang: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, lang)


def english_guard(src: str, out: str) -> str:
    """Undo accents put on a word that is plainly English and not Hungarian.
    Works per letter-run, so in 'deadline-ra' only 'deadline' is protected."""
    res = list(out)
    for m in WORD_RE.finditer(src):
        w_in, w_out = m.group(), out[m.start() : m.end()]
        if w_in == w_out:
            continue
        lo = w_in.lower()
        if _zipf(lo, "en") >= 3.0 and _zipf(w_out.lower(), "hu") < 1.5 and _zipf(lo, "hu") < _zipf(lo, "en"):
            res[m.start() : m.end()] = list(w_in)
    return "".join(res)


def _match_case(form: str, original: str) -> str:
    return "".join(f.upper() if o.isupper() else f for f, o in zip(form, original))


class Restorer:
    def __init__(
        self,
        model: str | Path | AccentModel = DEFAULT_MODEL,
        device: str = "cpu",
        guard: bool = True,
        lexicon: str | Path | Lexicon | None = DEFAULT_LEXICON,
        alpha: float = 0.25,
        min_total: int = 0,
    ):
        self.device = torch.device(device)
        self.model = (model if isinstance(model, AccentModel) else AccentModel.load(model)).to(self.device)
        self.guard = guard
        if isinstance(lexicon, (str, Path)):
            lexicon = Lexicon.load(lexicon) if Path(lexicon).exists() else None
        self.lexicon = lexicon
        self.alpha = alpha
        self.min_total = min_total

    # ------------------------------------------------------------ model
    @torch.inference_mode()
    def _logits(self, texts: list[str], chunk: int = 384, ctx: int = 64) -> list[np.ndarray]:
        """Per-character logits for each text, using overlapping windows for long texts."""
        pieces, owners = [], []
        for ti, t in enumerate(texts):
            pos = 0
            while pos < len(t):
                end = min(len(t), pos + chunk)
                if end < len(t):
                    sp = t.rfind(" ", pos + chunk // 2, end)
                    end = sp if sp > 0 else end
                ws, we = max(0, pos - ctx), min(len(t), end + ctx)
                pieces.append(t[ws:we])
                owners.append((ti, pos, end, ws))
                pos = end
        out = [np.zeros((len(t), 4), dtype=np.float32) for t in texts]
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        for b in range(0, len(order), 64):
            idx = order[b : b + 64]
            T = max(len(pieces[i]) for i in idx)
            ids = torch.zeros(len(idx), T, dtype=torch.long)
            for r, i in enumerate(idx):
                e = encode_input(pieces[i])
                ids[r, : len(e)] = torch.from_numpy(e)
            logits = self.model(ids.to(self.device)).float().cpu().numpy()
            for r, i in enumerate(idx):
                ti, cs, ce, ws = owners[i]
                out[ti][cs:ce] = logits[r, cs - ws : ce - ws]
        return out

    @staticmethod
    def _logprobs(text: str, logits: np.ndarray) -> np.ndarray:
        lp = np.full(logits.shape, -np.inf, dtype=np.float32)
        for i, c in enumerate(text):
            if c in BASE_VOWELS:
                z = np.where(VALID_MASK[c], logits[i], -np.inf)
                lp[i] = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return lp

    # ------------------------------------------------------------ decoding
    def _decode(self, text: str, logits: np.ndarray, want_words: bool = False):
        lp = self._logprobs(text, logits)
        chars = list(text)
        for i, c in enumerate(text):
            if c in BASE_VOWELS:  # user-typed accents are kept as-is
                chars[i] = apply_label(c, int(np.argmax(lp[i])))
        words: list[Word] = []
        for m in TOKEN_RE.finditer(text):
            w, s, e = m.group(), m.start(), m.end()
            vowel_pos = [i for i in range(s, e) if text[i] in BASE_VOWELS]
            if not vowel_pos:
                continue
            low = w.lower()
            model_form = "".join(chars[s:e]).lower()
            cands, unseen = ({}, 0.0)
            if self.lexicon is not None and len(low) == len(w):
                cands, unseen = self.lexicon.candidates(low, self.min_total)
            # forms must agree with accents the user typed
            cands = {f: p for f, p in cands.items() if all(text[i] in BASE_VOWELS or f[i - s] == low[i - s]
                                                          for i in range(s, e))}
            if model_form not in cands:
                cands[model_form] = unseen

            def model_lp(f: str) -> float:
                return float(sum(lp[i][LABEL_OF[f[i - s]]] for i in vowel_pos))

            scored = sorted(((model_lp(f) + self.alpha * p, f) for f, p in cands.items()), reverse=True)
            best = scored[0][1]
            chars[s:e] = list(_match_case(best, w))
            if want_words:
                z = np.array([sc for sc, _ in scored])
                probs = np.exp(z - z.max())
                probs /= probs.sum()
                alts = [(_match_case(f, w), float(p)) for (_, f), p in zip(scored[1:], probs[1:]) if p > 0.02]
                words.append(Word(s, e, _match_case(best, w), float(probs[0]), alts))
        out = "".join(chars)
        if self.guard:
            out = english_guard(text, out)
        return out, words

    # ------------------------------------------------------------ API
    def restore_many(self, texts: list[str]) -> list[str]:
        texts = [normalize(t) for t in texts]
        return [self._decode(t, lg)[0] for t, lg in zip(texts, self._logits(texts))]

    def restore(self, text: str) -> str:
        return self.restore_many([text])[0]

    def analyze(self, text: str) -> tuple[str, list[Word]]:
        """Restored text plus per-word confidence and alternatives (for UI highlighting)."""
        text = normalize(text)
        out, words = self._decode(text, self._logits([text])[0], want_words=True)
        return out, [w for w in words if out[w.start : w.end] == w.text]
