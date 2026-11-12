"""Small bidirectional character-level Transformer encoder (RoPE, SwiGLU, RMSNorm).

Input: character ids. Output: 4 logits per position (which accent variant of the
base vowel). Non-vowel positions are ignored by the loss and by decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chars import NUM_CLASSES, VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    dim: int = 256
    layers: int = 6
    heads: int = 4
    ffn_mult: float = 2.75
    conv_kernel: int = 7
    dropout: float = 0.05
    max_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(head_dim: int, max_len: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):  # x: (B, H, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T], sin[:T]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.heads
        self.n1 = RMSNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.n2 = RMSNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.ffn_mult)
        self.w12 = nn.Linear(cfg.dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, attn_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        h1, h2 = self.w12(self.n2(x)).chunk(2, dim=-1)
        return x + self.drop(self.w3(F.silu(h1) * h2))


class AccentModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0)
        # Local spelling patterns (suffixes, letter clusters) via a depthwise conv stem.
        self.conv = nn.Conv1d(cfg.dim, cfg.dim, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=cfg.dim)
        self.conv_norm = RMSNorm(cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, NUM_CLASSES)
        cos, sin = rope_cache(cfg.dim // cfg.heads, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad = ids == 0
        x = self.emb(ids)
        x = x + self.conv(self.conv_norm(x).transpose(1, 2)).transpose(1, 2)
        # Boolean mask: True = may attend. Shape (B, 1, 1, T) broadcasts over heads/queries.
        attn_mask = (~pad)[:, None, None, :]
        for b in self.blocks:
            x = b(x, self.cos, self.sin, attn_mask)
        return self.head(self.norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint(self) -> dict:
        return {"config": asdict(self.cfg), "state": self.state_dict()}

    @classmethod
    def load(cls, path, map_location="cpu") -> "AccentModel":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(Config(**ck["config"]))
        m.load_state_dict(ck["state"])
        return m.eval()
