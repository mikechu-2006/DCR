"""Span masking on the 50 Hz feature grid (data2vec-audio style).

For every sample we draw int(mask_prob * T) start indices and mask
mask_length frames from each start. Starts may overlap, which gives the
~49% effective ratio reported for data2vec-audio (p=0.065, length=10).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def span_mask(valid_len: torch.Tensor, total_frames: int, mask_prob: float,
              mask_length: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return a bool mask (B, T); True = masked. Padding is never masked."""
    b = int(valid_len.numel())
    mask = torch.zeros((b, total_frames), dtype=torch.bool)
    for i in range(b):
        n = int(valid_len[i].item())
        if n <= 0:
            continue
        num = max(1, int(mask_prob * n))
        starts = torch.randint(0, n, (num,), generator=generator)
        for s in starts.tolist():
            e = min(n, s + int(mask_length))
            mask[i, s:e] = True
    return mask


def mask_ratio(mask: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(mask.size(1), device=mask.device)[None, :] < valid_len[:, None]
    denom = valid.sum().clamp_min(1)
    return (mask & valid).sum().float() / denom.float()
