from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

PROPOSAL_ARCHITECTURE_ID = "high_low_rectified_lpjepa_sae_v2_axis_rdm"
SUPPORTED_RGG_P = (1.0, 2.0)


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


def unit_variance_generalized_gaussian_sigma(p: float) -> float:
    if p not in SUPPORTED_RGG_P:
        raise ValueError(f"p must be one of {SUPPORTED_RGG_P}")
    return math.sqrt(math.gamma(1.0 / p)) / (
        p ** (1.0 / p) * math.sqrt(math.gamma(3.0 / p))
    )


def rgg_mean_for_active_fraction(
    p: float, active_fraction: float, sigma: float
) -> float:
    if not 0.0 < active_fraction < 1.0:
        raise ValueError("active_fraction must lie strictly between zero and one")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if p == 1.0:
        if active_fraction <= 0.5:
            return sigma * math.log(2.0 * active_fraction)
        return -sigma * math.log(2.0 * (1.0 - active_fraction))
    if p == 2.0:
        probability = torch.tensor(active_fraction, dtype=torch.float64)
        return float(
            sigma * torch.distributions.Normal(0.0, 1.0).icdf(probability).item()
        )
    raise ValueError(f"p must be one of {SUPPORTED_RGG_P}")


def sample_rectified_generalized_gaussian(
    shape: tuple[int, ...],
    *,
    p: float,
    mu: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if p == 1.0:
        uniform = torch.rand(shape, device=device, dtype=torch.float32) - 0.5
        noise = -sigma * uniform.sign() * torch.log1p(-2.0 * uniform.abs())
    elif p == 2.0:
        noise = sigma * torch.randn(shape, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"p must be one of {SUPPORTED_RGG_P}")
    return torch.relu(noise.add(mu)).to(dtype=dtype)


@dataclass
class RectifiedLpJEPAConfig:
    d_in: int
    d_sae: int
    low_k: int
    max_span_length: int
    high_fraction: float = 0.2
    high_reconstruction_weight: float = 0.1
    rgg_p: float = 1.0
    target_active_fraction: float = 0.025
    target_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.d_in < 1 or self.d_sae < 2:
            raise ValueError("d_in must be positive and d_sae must be at least two")
        if self.max_span_length < 2:
            raise ValueError("max_span_length must be at least two")
        if not 0 < self.high_fraction < 1:
            raise ValueError("high_fraction must lie strictly between zero and one")
        if not 0 <= self.high_reconstruction_weight <= 1:
            raise ValueError("high_reconstruction_weight must lie in [0, 1]")
        if self.d_high < 1 or self.d_low < 1:
            raise ValueError("both high and low dictionaries must be non-empty")
        if not 1 <= self.low_k <= self.d_low:
            raise ValueError("low_k must lie in [1, d_low]")
        if self.rgg_p not in SUPPORTED_RGG_P:
            raise ValueError(f"rgg_p must be one of {SUPPORTED_RGG_P}")
        if not 0.0 < self.target_active_fraction < 1.0:
            raise ValueError("target_active_fraction must lie in (0, 1)")
        if self.target_sigma < 0:
            raise ValueError("target_sigma must be zero (automatic) or positive")

    @property
    def d_high(self) -> int:
        return max(1, min(self.d_sae - 1, round(self.d_sae * self.high_fraction)))

    @property
    def d_low(self) -> int:
        return self.d_sae - self.d_high

    @property
    def high_size(self) -> int:
        return self.d_high

    @property
    def resolved_target_sigma(self) -> float:
        return self.target_sigma or unit_variance_generalized_gaussian_sigma(
            self.rgg_p
        )

    @property
    def target_mu(self) -> float:
        return rgg_mean_for_active_fraction(
            self.rgg_p, self.target_active_fraction, self.resolved_target_sigma
        )

    @property
    def expected_high_l0(self) -> float:
        return self.d_high * self.target_active_fraction


class RectifiedLpJEPASAE(nn.Module):
    """Predictor-free high/low SAE with an RGG-regularized shared high code."""

    def __init__(self, cfg: RectifiedLpJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("pre_scale", torch.ones(()))
        self.encoder = nn.Linear(cfg.d_in, cfg.d_sae)
        self.decoder = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        nn.init.kaiming_uniform_(self.decoder, a=math.sqrt(5))
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()
            self.encoder.bias[: cfg.d_high].fill_(cfg.target_mu)

    @torch.no_grad()
    def initialize_normalization(self, mean: torch.Tensor, scalar_rms: float) -> None:
        if mean.shape != self.pre_bias.shape:
            raise ValueError("normalization mean does not match residual width")
        self.pre_bias.copy_(mean.to(self.pre_bias))
        self.pre_scale.copy_(torch.as_tensor(scalar_rms).to(self.pre_scale))
        self.normalize_decoder()
        self.encoder.weight.copy_(self.decoder)
        self.encoder.bias.zero_()
        self.encoder.bias[: self.cfg.d_high].fill_(self.cfg.target_mu)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def split_code(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return code[..., : self.cfg.d_high], code[..., self.cfg.d_high :]

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.pre_bias) / self.pre_scale)

    def encode(self, x: torch.Tensor, method: str = "proposal") -> torch.Tensor:
        del method
        dense = self.preactivations(x)
        high = F.relu(dense[..., : self.cfg.d_high])
        low = token_topk(dense[..., self.cfg.d_high :], self.cfg.low_k)
        return torch.cat((high, low), dim=-1)

    def decode(self, z: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        value = self.pre_scale * (z @ self.decoder)
        return value + self.pre_bias if add_bias else value

    def decode_high(
        self,
        z_high: torch.Tensor,
        *,
        add_bias: bool = True,
    ) -> torch.Tensor:
        value = self.pre_scale * (z_high @ self.decoder[: self.cfg.d_high])
        return value + self.pre_bias if add_bias else value

    def decode_low(
        self,
        z_low: torch.Tensor,
        *,
        add_bias: bool = False,
    ) -> torch.Tensor:
        value = self.pre_scale * (z_low @ self.decoder[self.cfg.d_high :])
        return value + self.pre_bias if add_bias else value

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if view_a.ndim != 2 or view_a.shape[-1] != self.cfg.d_in:
            raise ValueError("view_a must have shape [batch, d_in]")
        if view_b.shape != view_a.shape:
            raise ValueError("view_b must match view_a")
        code_a = self.encode(view_a)
        code_b = self.encode(view_b)
        high_a, low_a = self.split_code(code_a)
        high_b, low_b = self.split_code(code_b)
        high_reconstruction_a = self.decode_high(high_a)
        high_reconstruction_b = self.decode_high(high_b)
        full_reconstruction_a = high_reconstruction_a + self.decode_low(low_a)
        full_reconstruction_b = high_reconstruction_b + self.decode_low(low_b)
        return {
            "code_a": code_a,
            "code_b": code_b,
            "high_a": high_a,
            "high_b": high_b,
            "low_a": low_a,
            "low_b": low_b,
            "high_reconstruction_a": high_reconstruction_a,
            "high_reconstruction_b": high_reconstruction_b,
            "full_reconstruction_a": full_reconstruction_a,
            "full_reconstruction_b": full_reconstruction_b,
        }
