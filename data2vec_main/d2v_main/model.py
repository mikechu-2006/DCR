"""Audio-style data2vec model: conv feature encoder + student/EMA teacher.

    piano roll (B, 8, 100 Hz)
        -> Conv1d stack (stride 2) -> 50 Hz x dim
        -> + sinusoidal positions
        -> student: span-mask + Transformer
        -> teacher: no mask + EMA Transformer (shared conv)
        -> Smooth-L1 on the masked frames against top-K teacher layers
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence {x.size(1)} > max_frames {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class ConvFeatureEncoder(nn.Module):
    """100 Hz piano roll -> 50 Hz continuous features (stride 2)."""

    def __init__(self, in_channels: int = 8, hidden: int = 256, dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden, dim, kernel_size=5, stride=2, padding=2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        return self.norm(x.transpose(1, 2))


class TransformerStack(nn.Module):
    """Pre-norm Transformer; returns the final output and every layer output."""

    def __init__(self, dim: int, layers: int, heads: int, ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                return_layers: bool = True):
        outs = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
            if return_layers:
                outs.append(x)
        return self.norm(x), outs


class Data2VecModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        from .attributes import ChannelSpec
        spec = ChannelSpec.from_config(cfg)
        if int(m["in_channels"]) != spec.n_channels:
            raise ValueError(
                "model.in_channels=%r != data.channels n_channels=%d"
                % (m["in_channels"], spec.n_channels))
        self.dim = int(m["dim"])
        self.top_k = int(m["top_k"])
        self.loss_beta = float(m["loss_beta"])
        self.conv = ConvFeatureEncoder(int(m["in_channels"]), int(m["conv_hidden"]), self.dim)
        self.pos = SinusoidalPositionalEncoding(self.dim, int(m["max_frames"]))
        self.student = TransformerStack(
            self.dim, int(m["layers"]), int(m["heads"]), int(m["ff_mult"]), float(m["dropout"]))
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student_head = nn.Linear(self.dim, self.dim)
        self.mask_emb = nn.Parameter(torch.randn(self.dim) * 0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        f = self.conv(x)
        # Mask first, then add positions: masked frames must keep their time
        # identity (data2vec-audio's conv_pos is applied after masking too).
        if mask is not None:
            f = torch.where(mask.unsqueeze(-1), self.mask_emb, f)
        f = self.pos(f)
        return f

    def forward_student(self, x: torch.Tensor, valid_mask: torch.Tensor,
                        mask: torch.Tensor | None = None):
        f = self._features(x, mask)
        return self.student(f, key_padding_mask=~valid_mask)

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor, valid_mask: torch.Tensor):
        self.teacher.eval()
        f = self._features(x, None)
        return self.teacher(f, key_padding_mask=~valid_mask)

    def target(self, layers: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        k = min(self.top_k, len(layers))
        ys = [F.layer_norm(layer, (self.dim,)) for layer in layers[-k:]]
        y = torch.stack(ys, dim=0).mean(dim=0)
        return y[mask]

    def predict(self, out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.student_head(out[mask])

    def parameter_count(self) -> dict[str, int]:
        def n(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "conv": n(self.conv),
            "student": n(self.student),
            "teacher": n(self.teacher),
            "head": n(self.student_head),
            "total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class EMAState:
    """FP32 master copy of the teacher for stable EMA under bf16.

    The teacher module stays in the student dtype so its forward pass is fast;
    the master is updated in fp32 and copied into the teacher after every step.
    """

    def __init__(self, teacher: nn.Module, enabled: bool = True):
        self.enabled = bool(enabled)
        self.device = next(teacher.parameters()).device
        self.master = ([p.detach().float().clone() for p in teacher.parameters()]
                       if self.enabled else None)

    @torch.no_grad()
    def update(self, teacher: nn.Module, student: nn.Module, decay: float) -> None:
        if self.enabled:
            for m, t, s in zip(self.master, teacher.parameters(), student.parameters()):
                m.mul_(decay).add_(s.detach().float(), alpha=1.0 - decay)
                t.copy_(m)
        else:
            for t, s in zip(teacher.parameters(), student.parameters()):
                t.mul_(decay).add_(s.detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "master": self.master}

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.enabled = bool(state.get("enabled", self.enabled))
        master = state.get("master")
        if master is not None:
            self.master = [m.detach().float().to(self.device) for m in master]


def cosine_lr(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
