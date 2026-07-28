from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def token_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    positive = F.relu(values)
    selected = positive.topk(k, dim=-1, sorted=False)
    return torch.zeros_like(positive).scatter_(-1, selected.indices, selected.values)


def batch_topk(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("batch_topk expects [batch, features]")
    positive = F.relu(values)
    count = min(k * len(positive), positive.numel())
    selected = positive.flatten().topk(count, sorted=False)
    encoded = (
        torch.zeros_like(positive)
        .flatten()
        .scatter_(0, selected.indices, selected.values)
    )
    return encoded.reshape_as(positive), selected.values.min().detach()


@dataclass
class SparseAutoencoderConfig:
    d_in: int
    d_sae: int
    k: int
    high_fraction: float = 0.2

    @property
    def high_size(self) -> int:
        return max(1, min(self.d_sae - 1, int(self.d_sae * self.high_fraction)))


class SparseAutoencoder(nn.Module):
    """Shared dictionary used by the standard and temporal conditions."""

    def __init__(self, cfg: SparseAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("pre_scale", torch.ones(()))
        self.encoder = nn.Linear(cfg.d_in, cfg.d_sae)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        self.register_buffer("threshold", torch.tensor(-1.0))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()

    @torch.no_grad()
    def initialize_normalization(self, mean: torch.Tensor, scalar_rms: float) -> None:
        self.pre_bias.copy_(mean.to(self.pre_bias))
        self.pre_scale.copy_(torch.as_tensor(scalar_rms, device=self.pre_scale.device))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.pre_bias) / self.pre_scale)

    def encode_token_topk(self, x: torch.Tensor) -> torch.Tensor:
        return token_topk(self.preactivations(x), self.cfg.k)

    def encode_batch_topk(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.preactivations(x)
        code, minimum = batch_topk(pre, self.cfg.k)
        return code, minimum, F.relu(pre)

    def encode_threshold(self, x: torch.Tensor) -> torch.Tensor:
        positive = F.relu(self.preactivations(x))
        return positive * (positive > self.threshold)

    def encode(self, x: torch.Tensor, method: str) -> torch.Tensor:
        if method == "temporal":
            return self.encode_threshold(x)
        return self.encode_token_topk(x)

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.decoder)
        return value + self.pre_bias if add_bias else value

    def checkpoint_config(self) -> dict[str, int | float]:
        return asdict(self.cfg)


@dataclass
class TransitionJEPAConfig:
    d_in: int
    d_sae: int
    k: int
    window_size: int
    high_fraction: float = 0.2
    predictor_width: int = 256
    predictor_expansion: int = 2
    ema_decay: float = 0.996


class OffsetConditionedPredictor(nn.Module):
    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        width = cfg.predictor_width
        hidden = cfg.predictor_expansion * width
        self.context_projection = nn.Linear(cfg.d_sae, width, bias=False)
        self.offset_embedding = nn.Embedding(cfg.window_size, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, cfg.d_sae)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(
        self, context_code: torch.Tensor, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.context_projection(context_code)
        queries = state[:, None, :] + self.offset_embedding(offsets)[None]
        return F.softplus(self.output(self.mlp(queries))), state


class TransitionJEPA(nn.Module):
    """Offset-conditioned Transition JEPA-SAE from fumin0ri/my-sae."""

    def __init__(self, cfg: TransitionJEPAConfig, initialized_sae: SparseAutoencoder):
        super().__init__()
        self.cfg = cfg
        self.sae = copy.deepcopy(initialized_sae)
        self.target_encoder = copy.deepcopy(self.sae.encoder)
        self.predictor = OffsetConditionedPredictor(cfg)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def set_sae_trainable(self, trainable: bool) -> None:
        self.sae.pre_bias.requires_grad_(trainable)
        self.sae.decoder.requires_grad_(trainable)
        for parameter in self.sae.encoder.parameters():
            parameter.requires_grad_(trainable)

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(
            self.target_encoder.parameters(), self.sae.encoder.parameters()
        ):
            target.mul_(self.cfg.ema_decay).add_(
                online.detach(), alpha=1 - self.cfg.ema_decay
            )

    @torch.no_grad()
    def target_codes(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.sae.pre_bias.detach()) / self.sae.pre_scale
        return token_topk(self.target_encoder(normalized), self.cfg.k)

    def forward(self, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        if windows.ndim != 3 or windows.shape[1] != self.cfg.window_size:
            raise ValueError(f"expected [batch, {self.cfg.window_size}, d_in] windows")
        codes = self.sae.encode_token_topk(windows)
        reconstruction = self.sae.decode(codes)
        targets = self.target_codes(windows[:, 1:])
        offsets = torch.arange(1, self.cfg.window_size, device=windows.device)
        prediction, state = self.predictor(codes[:, 0], offsets)
        sparse_prediction = token_topk(prediction, self.cfg.k)
        return {
            "codes": codes,
            "reconstruction": reconstruction,
            "targets": targets,
            "prediction": prediction,
            "sparse_prediction": sparse_prediction,
            "predicted_residual": self.sae.decode(sparse_prediction),
            "state": state,
        }
