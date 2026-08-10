"""Reusable neural layers for quotient extraction."""

from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class GradientReversal(nn.Module):
    """Identity in the forward pass and sign-reversed gradient in backward."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(value, self.scale)


class WorkspaceGate(nn.Module):
    """Learned sparse gate over model-local capture coordinates."""

    def __init__(self, input_dim: int, enabled: bool):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        if enabled:
            self.logits = nn.Parameter(torch.full((input_dim,), 1.5))
        else:
            self.register_parameter("logits", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.logits is None:
            return value
        return value * torch.sigmoid(self.logits)

    def regularization(self) -> torch.Tensor:
        if not self.enabled or self.logits is None:
            return self._zero
        return torch.sigmoid(self.logits).mean()

    def active_fraction(self, threshold: float = 0.5) -> float:
        if not self.enabled or self.logits is None:
            return 1.0
        return float((torch.sigmoid(self.logits.detach()) >= threshold).float().mean().item())


class StraightThroughQuantizer(nn.Module):
    """Uniform bounded quantization with a straight-through estimator."""

    def __init__(self, bits: int):
        super().__init__()
        if not 1 <= bits <= 16:
            raise ValueError("bits must be between 1 and 16")
        self.bits = bits
        self.levels = 2**bits - 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(value)
        scaled = (bounded + 1.0) * 0.5 * self.levels
        quantized = torch.round(scaled) / self.levels * 2.0 - 1.0
        return bounded + (quantized - bounded).detach()

    def hard_quantize(self, value: torch.Tensor, bits: int | None = None) -> torch.Tensor:
        selected_bits = self.bits if bits is None else bits
        levels = 2**selected_bits - 1
        bounded = torch.clamp(value, -1.0, 1.0)
        return torch.round((bounded + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0


class PublicOperationDecoder(nn.Module):
    """Execute public future-operation coefficients on a decoded causal state.

    This decoder has no learned parameters. It is the synthetic Stage-0
    analogue of a frozen interrogator: held-out operations are executable as
    soon as their public coefficient descriptors are supplied.
    """

    def __init__(self, n_facts: int, n_residual: int, descriptor_dim: int):
        super().__init__()
        self.n_facts = n_facts
        self.n_residual = n_residual
        self.pair_count = n_facts * (n_facts - 1) // 2
        self.basis_dim = n_facts + self.pair_count + n_residual + 1
        self.coefficient_offset = descriptor_dim - self.basis_dim
        if self.coefficient_offset < 1:
            raise ValueError("operation descriptor does not contain a public coefficient block")

    def forward(
        self,
        fact_logits: torch.Tensor,
        residual: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        fact_probabilities = torch.sigmoid(fact_logits)
        signed = 2.0 * fact_probabilities - 1.0
        pair_terms = [
            signed[:, i] * signed[:, j]
            for i in range(self.n_facts)
            for j in range(i + 1, self.n_facts)
        ]
        if pair_terms:
            pairs = torch.stack(pair_terms, dim=1)
        else:
            pairs = signed.new_zeros((signed.shape[0], 0))
        constant = signed.new_ones((signed.shape[0], 1))
        basis = torch.cat([signed, pairs, residual, constant], dim=1)
        coefficients = descriptors[:, self.coefficient_offset :]
        if coefficients.shape[1] != basis.shape[1]:
            raise ValueError("public operation coefficient width does not match decoded state")
        return basis @ coefficients.transpose(0, 1)


class DotProductOperationDecoder(nn.Module):
    """Decode future operations from one operation-agnostic state code."""

    def __init__(
        self,
        code_dim: int,
        operation_descriptor_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.code_projection = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Linear(code_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.operation_projection = nn.Sequential(
            nn.LayerNorm(operation_descriptor_dim),
            nn.Linear(operation_descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.operation_bias = nn.Sequential(
            nn.Linear(operation_descriptor_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, code: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        code_features = self.code_projection(code)
        operation_features = self.operation_projection(descriptors)
        logits = code_features @ operation_features.transpose(0, 1) * self.scale
        return logits + self.operation_bias(descriptors).transpose(0, 1)
