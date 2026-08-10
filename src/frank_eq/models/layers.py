"""Reusable neural layers for quotient extraction."""

from __future__ import annotations

import math

import torch
from torch import nn

from frank_eq.real_config import GRAPH_OPERATION_FAMILIES


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
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(value, self.scale)


class WorkspaceGate(nn.Module):
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
    """Execute public synthetic coefficients on a decoded causal state."""

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
        pairs = (
            torch.stack(pair_terms, dim=1)
            if pair_terms
            else signed.new_zeros((signed.shape[0], 0))
        )
        constant = signed.new_ones((signed.shape[0], 1))
        basis = torch.cat([signed, pairs, residual, constant], dim=1)
        coefficients = descriptors[:, self.coefficient_offset :]
        if coefficients.shape[1] != basis.shape[1]:
            raise ValueError("public operation coefficient width does not match decoded state")
        return basis @ coefficients.transpose(0, 1)


class GraphOperationDecoder(nn.Module):
    """Frozen differentiable interrogator for the real relational-world panel."""

    def __init__(
        self,
        *,
        n_entities: int,
        n_residual: int,
        descriptor_dim: int,
        temperature: float,
    ):
        super().__init__()
        if n_entities < 3:
            raise ValueError("GraphOperationDecoder requires at least three entities")
        if n_residual < 2:
            raise ValueError("GraphOperationDecoder requires density and reciprocity residuals")
        expected = len(GRAPH_OPERATION_FAMILIES) + 4 * n_entities + 1
        if descriptor_dim != expected:
            raise ValueError(
                f"graph operation descriptor width must be {expected}, got {descriptor_dim}"
            )
        self.n_entities = n_entities
        self.n_facts = n_entities * (n_entities - 1)
        self.n_residual = n_residual
        self.temperature = float(temperature)
        edge_index = torch.full((n_entities, n_entities), -1, dtype=torch.long)
        cursor = 0
        for source in range(n_entities):
            for target in range(n_entities):
                if source == target:
                    continue
                edge_index[source, target] = cursor
                cursor += 1
        self.register_buffer("edge_index", edge_index, persistent=True)

    def _edge(self, probabilities: torch.Tensor, source: int, target: int) -> torch.Tensor:
        if source == target:
            return probabilities.new_zeros(probabilities.shape[0])
        index = int(self.edge_index[source, target].item())
        return probabilities[:, index]

    def _compose(
        self,
        probabilities: torch.Tensor,
        source: int,
        target: int,
        *,
        forced_edge: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        path_probabilities: list[torch.Tensor] = []
        for middle in range(self.n_entities):
            if middle in {source, target}:
                continue
            left = self._edge(probabilities, source, middle)
            right = self._edge(probabilities, middle, target)
            if forced_edge == (source, middle):
                left = torch.ones_like(left)
            if forced_edge == (middle, target):
                right = torch.ones_like(right)
            path_probabilities.append(left * right)
        if not path_probabilities:
            return probabilities.new_zeros(probabilities.shape[0])
        paths = torch.stack(path_probabilities, dim=1)
        return 1.0 - torch.prod(1.0 - paths, dim=1)

    @staticmethod
    def _bounded_logit(probability: torch.Tensor) -> torch.Tensor:
        return torch.logit(torch.clamp(probability, 1e-5, 1.0 - 1e-5))

    def forward(
        self,
        fact_logits: torch.Tensor,
        residual: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(fact_logits)
        if probabilities.shape[1] != self.n_facts:
            raise ValueError(
                f"graph decoder expected {self.n_facts} edge facts, got {probabilities.shape[1]}"
            )
        family_width = len(GRAPH_OPERATION_FAMILIES)
        family_ids = torch.argmax(descriptors[:, :family_width], dim=1)
        offset = family_width
        argument_blocks = [
            torch.argmax(
                descriptors[:, offset + block * self.n_entities : offset + (block + 1) * self.n_entities],
                dim=1,
            )
            for block in range(4)
        ]
        polarities = descriptors[:, -1]
        outputs: list[torch.Tensor] = []
        for operation_index in range(descriptors.shape[0]):
            family = GRAPH_OPERATION_FAMILIES[int(family_ids[operation_index].item())]
            source, target, aux_source, aux_target = (
                int(block[operation_index].item()) for block in argument_blocks
            )
            if family == "lookup":
                probability = self._edge(probabilities, source, target)
            elif family == "inverse":
                probability = self._edge(probabilities, target, source)
            elif family == "mutual":
                probability = self._edge(probabilities, source, target) * self._edge(
                    probabilities, target, source
                )
            elif family == "compose":
                probability = self._compose(probabilities, source, target)
            elif family == "compare_outdegree":
                source_degree = torch.stack(
                    [
                        self._edge(probabilities, source, other)
                        for other in range(self.n_entities)
                        if other != source
                    ],
                    dim=1,
                ).sum(dim=1)
                target_degree = torch.stack(
                    [
                        self._edge(probabilities, target, other)
                        for other in range(self.n_entities)
                        if other != target
                    ],
                    dim=1,
                ).sum(dim=1)
                probability = torch.sigmoid(
                    self.temperature * (source_degree - target_degree - 0.5)
                )
            elif family == "counterfactual_add":
                probability = self._compose(
                    probabilities,
                    aux_source,
                    aux_target,
                    forced_edge=(source, target),
                )
            elif family == "density":
                probability = torch.sigmoid(self.temperature * (residual[:, 0] - 1e-4))
            elif family == "reciprocity":
                probability = torch.sigmoid(self.temperature * (residual[:, 1] - 1e-4))
            else:
                raise ValueError(f"unsupported graph operation family: {family}")
            if float(polarities[operation_index].item()) < 0:
                probability = 1.0 - probability
            outputs.append(self._bounded_logit(probability))
        return torch.stack(outputs, dim=1)


class DotProductOperationDecoder(nn.Module):
    """Optional learned comparison decoder; not used by the primary Stage-A path."""

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
