from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

PROPOSAL_ARCHITECTURE_ID = "high_low_random_pair_horizon_ema_sae_v3"


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
    group_topk: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.high_fraction < 1:
            raise ValueError("high_fraction must lie strictly between zero and one")
        if self.group_topk and self.k < 2:
            raise ValueError("group_topk requires k >= 2")

    @property
    def high_size(self) -> int:
        return max(1, min(self.d_sae - 1, int(self.d_sae * self.high_fraction)))

    @property
    def group_high_size(self) -> int:
        return max(1, min(self.d_sae - 1, round(self.d_sae * self.high_fraction)))

    @property
    def group_high_k(self) -> int:
        proposed = round(self.k * self.high_fraction)
        return max(1, min(self.group_high_size, self.k - 1, proposed))

    @property
    def group_low_k(self) -> int:
        return self.k - self.group_high_k


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

    def sparsify_token(self, preactivations: torch.Tensor) -> torch.Tensor:
        if not self.cfg.group_topk:
            return token_topk(preactivations, self.cfg.k)
        high_size = self.cfg.group_high_size
        high = token_topk(
            preactivations[..., :high_size], self.cfg.group_high_k
        )
        low = token_topk(
            preactivations[..., high_size:], self.cfg.group_low_k
        )
        return torch.cat((high, low), dim=-1)

    def encode_token_topk(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsify_token(self.preactivations(x))

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
    high_reconstruction_weight: float = 0.2
    predictor_width: int = 256
    predictor_expansion: int = 2
    ema_decay: float = 0.996

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size/max_span_length must be at least two")
        if not 0 < self.high_fraction < 1:
            raise ValueError("high_fraction must lie strictly between zero and one")
        if not 0 <= self.high_reconstruction_weight <= 1:
            raise ValueError("high_reconstruction_weight must lie in [0, 1]")
        if self.d_high < 1 or self.d_low < 1:
            raise ValueError("both high and low dictionaries must be non-empty")
        if self.k < 2 or self.k_high < 1 or self.k_low < 1:
            raise ValueError("both high and low Top-K budgets must be positive")
        if self.k > self.d_sae:
            raise ValueError("k must not exceed d_sae")

    @property
    def d_high(self) -> int:
        return max(1, min(self.d_sae - 1, round(self.d_sae * self.high_fraction)))

    @property
    def d_low(self) -> int:
        return self.d_sae - self.d_high

    @property
    def k_high(self) -> int:
        proposed = round(self.k * self.high_fraction)
        return max(1, min(self.d_high, self.k - 1, proposed))

    @property
    def k_low(self) -> int:
        return self.k - self.k_high


class HorizonConditionedPredictor(nn.Module):
    """Predict a future high code from context and explicit token distance."""

    def __init__(self, cfg: TransitionJEPAConfig, feature_dim: int):
        super().__init__()
        width = cfg.predictor_width
        hidden = cfg.predictor_expansion * width
        self.context_projection = nn.Linear(feature_dim, width, bias=False)
        self.horizon_embedding = nn.Embedding(cfg.window_size, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, feature_dim)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(
        self,
        context_code: torch.Tensor,
        horizons: torch.Tensor,
        use_context: bool = True,
    ) -> torch.Tensor:
        squeeze_context = context_code.ndim == 2
        if squeeze_context:
            context_code = context_code[:, None, :]
        if context_code.ndim != 3:
            raise ValueError("context_code must have shape [batch, contexts, d_sae]")
        if torch.any(horizons < 1) or torch.any(horizons >= self.horizon_embedding.num_embeddings):
            raise ValueError("horizons must lie in [1, max_span_length-1]")
        if squeeze_context and horizons.shape == (context_code.shape[0],):
            horizon_state = self.horizon_embedding(horizons)[:, None, :]
        elif (
            not squeeze_context
            and horizons.ndim == 1
            and horizons.shape == (context_code.shape[1],)
        ):
            horizon_state = self.horizon_embedding(horizons)[None, :, :]
        elif horizons.shape == context_code.shape[:2]:
            horizon_state = self.horizon_embedding(horizons)
        else:
            raise ValueError("horizons must match the batch or context axis")
        if use_context:
            state = self.context_projection(context_code)
        else:
            state = torch.zeros(
                (*context_code.shape[:-1], self.context_projection.out_features),
                device=context_code.device,
                dtype=context_code.dtype,
            )
        output = F.softplus(self.output(self.mlp(state + horizon_state)))
        return output[:, 0] if squeeze_context else output


class TransitionJEPA(nn.Module):
    """High/low SAE trained from random context/endpoint/horizon pairs."""

    def __init__(self, cfg: TransitionJEPAConfig, initialized_sae: SparseAutoencoder):
        super().__init__()
        self.cfg = cfg
        self.sae = copy.deepcopy(initialized_sae)
        self.sae.cfg.group_topk = True
        self.sae.cfg.high_fraction = cfg.high_fraction
        self.ema_encoder = copy.deepcopy(self.sae.encoder)
        self.ema_decoder = nn.Parameter(
            self.sae.decoder.detach().clone(),
            requires_grad=False,
        )
        self.register_buffer("ema_pre_bias", self.sae.pre_bias.detach().clone())
        self.predictor = HorizonConditionedPredictor(cfg, cfg.d_high)
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
        return self.sae.sparsify_token(self.ema_encoder(normalized))

    def decode_ema(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        """Decode with the frozen EMA dictionary while preserving dz gradients."""
        value = self.sae.pre_scale * (z @ self.ema_decoder)
        return value + self.ema_pre_bias if add_bias else value

    def split_code(
        self, code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return code[..., : self.cfg.d_high], code[..., self.cfg.d_high :]

    def decode_high(
        self,
        z_high: torch.Tensor,
        *,
        ema: bool,
        add_bias: bool = True,
    ) -> torch.Tensor:
        decoder = self.ema_decoder if ema else self.sae.decoder
        bias = self.ema_pre_bias if ema else self.sae.pre_bias
        value = self.sae.pre_scale * (z_high @ decoder[: self.cfg.d_high])
        return value + bias if add_bias else value

    def decode_low(
        self,
        z_low: torch.Tensor,
        *,
        ema: bool,
        add_bias: bool = False,
    ) -> torch.Tensor:
        decoder = self.ema_decoder if ema else self.sae.decoder
        bias = self.ema_pre_bias if ema else self.sae.pre_bias
        value = self.sae.pre_scale * (z_low @ decoder[self.cfg.d_high :])
        return value + bias if add_bias else value

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
        horizons: torch.Tensor,
        *,
        use_context: bool = True,
        sparse_output: bool = False,
    ) -> torch.Tensor:
        if context_code.shape[-1] == self.cfg.d_sae:
            context_code = context_code[..., : self.cfg.d_high]
        if context_code.shape[-1] != self.cfg.d_high:
            raise ValueError(
                f"context code must have width {self.cfg.d_high} (high) or "
                f"{self.cfg.d_sae} (full)"
            )
        dense = self.predictor(
            context_code,
            horizons.to(device=context_code.device, dtype=torch.long),
            use_context=use_context,
        )
        return token_topk(dense, self.cfg.k_high) if sparse_output else dense

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        horizon: torch.Tensor,
        *,
        use_context: bool = True,
        use_ema_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 2 or context.shape[-1] != self.cfg.d_in:
            raise ValueError("context must have shape [batch, d_in]")
        if target.shape != context.shape:
            raise ValueError("target must match context shape")
        if horizon.shape != (len(context),):
            raise ValueError("horizon must have shape [batch]")
        target_codes_online = self.sae.encode_token_topk(target)
        online_target_code, online_target_low_code = self.split_code(
            target_codes_online
        )
        if use_ema_context:
            with torch.no_grad():
                context_full_code = self.encode_ema(context)
        else:
            context_full_code = self.sae.encode_token_topk(context)
        context_code, low_context_code = self.split_code(context_full_code)
        online_high_reconstruction = self.decode_high(
            online_target_code, ema=False
        )
        online_target_reconstruction = online_high_reconstruction + self.decode_low(
            online_target_low_code, ema=False, add_bias=False
        )
        with torch.no_grad():
            target_full_code = self.encode_ema(target)
            target_code, target_low_code = self.split_code(target_full_code)
            target_high_reconstruction = self.decode_high(target_code, ema=True)
            target_reconstruction = target_high_reconstruction + self.decode_low(
                target_low_code, ema=True, add_bias=False
            )
        predicted_codes = self.predict_from_code(
            context_code, horizon, use_context=use_context
        )
        sparse_prediction = token_topk(predicted_codes, self.cfg.k_high)
        predictable_residual = self.decode_high(sparse_prediction, ema=True)
        return {
            "codes": target_codes_online,
            "high_codes": online_target_code,
            "low_codes": online_target_low_code,
            "context_codes": context_code,
            "context_code": context_code,
            "low_context_codes": low_context_code,
            "low_context_code": low_context_code,
            "online_target_code": online_target_code,
            "online_target_low_code": online_target_low_code,
            "online_target_full_code": target_codes_online,
            "online_high_reconstruction": online_high_reconstruction,
            "online_target_reconstruction": online_target_reconstruction,
            "target_high_reconstruction": target_high_reconstruction,
            "target_reconstruction": target_reconstruction,
            "target_code": target_code,
            "target_low_code": target_low_code,
            "target_full_code": target_full_code,
            "target_codes": target_code,
            "predicted_codes": predicted_codes,
            "sparse_predicted_codes": sparse_prediction,
            "target_residual": target,
            "predictable_residual": predictable_residual,
            "innovation_residual": target - predictable_residual,
            "horizon": horizon,
        }
