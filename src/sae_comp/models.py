from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

PROPOSAL_ARCHITECTURE_ID = "all_context_fixed_endpoint_ema_sae_v2"


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


class PositionConditionedPredictor(nn.Module):
    """Predict the fixed endpoint code from each context code and position."""

    def __init__(self, cfg: TransitionJEPAConfig):
        super().__init__()
        width = cfg.predictor_width
        hidden = cfg.predictor_expansion * width
        self.context_projection = nn.Linear(cfg.d_sae, width, bias=False)
        self.position_embedding = nn.Embedding(cfg.window_size, width)
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
        self,
        context_code: torch.Tensor,
        context_positions: torch.Tensor,
        use_context: bool = True,
    ) -> torch.Tensor:
        if context_code.ndim == 2:
            context_code = context_code[:, None, :]
        if context_code.ndim != 3:
            raise ValueError("context_code must have shape [batch, contexts, d_sae]")
        if (
            context_positions.ndim != 1
            or len(context_positions) != context_code.shape[1]
        ):
            raise ValueError(
                "context_positions must contain one value for each context"
            )
        if use_context:
            state = self.context_projection(context_code)
        else:
            state = torch.zeros(
                (*context_code.shape[:-1], self.context_projection.out_features),
                device=context_code.device,
                dtype=context_code.dtype,
            )
        queries = state + self.position_embedding(context_positions)[None]
        return F.softplus(self.output(self.mlp(queries)))


class TransitionJEPA(nn.Module):
    """Forecast one fixed endpoint from every earlier residual in a window."""

    def __init__(self, cfg: TransitionJEPAConfig, initialized_sae: SparseAutoencoder):
        super().__init__()
        self.cfg = cfg
        self.sae = copy.deepcopy(initialized_sae)
        self.ema_encoder = copy.deepcopy(self.sae.encoder)
        self.ema_decoder = nn.Parameter(
            self.sae.decoder.detach().clone(),
            requires_grad=False,
        )
        self.register_buffer("ema_pre_bias", self.sae.pre_bias.detach().clone())
        self.predictor = PositionConditionedPredictor(cfg)
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad_(False)
        self.normalize_ema_decoder()

    def set_sae_trainable(self, trainable: bool) -> None:
        self.sae.pre_bias.requires_grad_(trainable)
        self.sae.decoder.requires_grad_(trainable)
        for parameter in self.sae.encoder.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad_(False)
        self.ema_decoder.requires_grad_(False)

    @torch.no_grad()
    def update_ema_sae(self, decay: float | None = None) -> None:
        rate = self.cfg.ema_decay if decay is None else decay
        for target, online in zip(
            self.ema_encoder.parameters(), self.sae.encoder.parameters()
        ):
            target.mul_(rate).add_(online.detach(), alpha=1 - rate)
        self.ema_pre_bias.mul_(rate).add_(
            self.sae.pre_bias.detach(), alpha=1 - rate
        )
        self.ema_decoder.mul_(rate).add_(
            self.sae.decoder.detach(), alpha=1 - rate
        )
        self.normalize_ema_decoder()

    @torch.no_grad()
    def normalize_ema_decoder(self) -> None:
        self.ema_decoder.div_(
            self.ema_decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )

    @torch.no_grad()
    def encode_ema(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.ema_pre_bias) / self.sae.pre_scale
        return token_topk(self.ema_encoder(normalized), self.cfg.k)

    def decode_ema(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        """Decode with the frozen EMA dictionary while preserving dz gradients."""
        value = self.sae.pre_scale * (z @ self.ema_decoder)
        return value + self.ema_pre_bias if add_bias else value

    @torch.no_grad()
    def final_ema_sae(self) -> SparseAutoencoder:
        """Export the full EMA teacher as the standalone SAE used downstream."""
        result = SparseAutoencoder(copy.deepcopy(self.sae.cfg)).to(
            device=self.ema_decoder.device,
            dtype=self.ema_decoder.dtype,
        )
        result.pre_bias.copy_(self.ema_pre_bias)
        result.pre_scale.copy_(self.sae.pre_scale)
        result.encoder.load_state_dict(self.ema_encoder.state_dict())
        result.decoder.copy_(self.ema_decoder)
        result.threshold.copy_(self.sae.threshold)
        return result

    def predict_from_code(
        self,
        context_code: torch.Tensor,
        context_positions: torch.Tensor | None = None,
        *,
        use_context: bool = True,
        sparse_output: bool = False,
    ) -> torch.Tensor:
        if context_code.ndim == 2:
            context_code = context_code[:, None, :]
        if context_positions is None:
            if context_code.shape[1] == self.cfg.window_size - 1:
                context_positions = torch.arange(
                    self.cfg.window_size - 1,
                    device=context_code.device,
                    dtype=torch.long,
                )
            elif context_code.shape[1] == 1:
                context_positions = torch.zeros(
                    1, device=context_code.device, dtype=torch.long
                )
            else:
                raise ValueError(
                    "explicit context_positions are required for this shape"
                )
        dense = self.predictor(
            context_code,
            context_positions.to(device=context_code.device, dtype=torch.long),
            use_context=use_context,
        )
        return token_topk(dense, self.cfg.k) if sparse_output else dense

    def forward(
        self,
        windows: torch.Tensor,
        *,
        use_context: bool = True,
        use_ema_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if windows.ndim != 3 or windows.shape[1] != self.cfg.window_size:
            raise ValueError(f"expected [batch, {self.cfg.window_size}, d_in] windows")
        codes = self.sae.encode_token_topk(windows)
        if use_ema_context:
            with torch.no_grad():
                context_codes = self.encode_ema(windows[:, :-1])
        else:
            context_codes = codes[:, :-1]
        online_target_code = codes[:, -1]
        online_target_reconstruction = self.sae.decode(online_target_code)
        with torch.no_grad():
            target_code = self.encode_ema(windows[:, -1])
            target_reconstruction = self.decode_ema(target_code)
        predicted_codes = self.predict_from_code(
            context_codes, use_context=use_context
        )
        sparse_prediction = token_topk(predicted_codes, self.cfg.k)
        predictable_residual = self.decode_ema(sparse_prediction)
        target_codes = target_code[:, None, :].expand_as(predicted_codes)
        target_residual = windows[:, -1][:, None, :].expand_as(
            predictable_residual
        )
        return {
            "codes": codes,
            "context_codes": context_codes,
            "context_code": context_codes[:, 0],
            "online_target_code": online_target_code,
            "online_target_reconstruction": online_target_reconstruction,
            "target_reconstruction": target_reconstruction,
            "target_code": target_code,
            "target_codes": target_codes,
            "predicted_codes": predicted_codes,
            "sparse_predicted_codes": sparse_prediction,
            "target_residual": target_residual,
            "predictable_residual": predictable_residual,
            "innovation_residual": target_residual - predictable_residual,
        }
