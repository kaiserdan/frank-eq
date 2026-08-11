"""Frozen architectural and text controls for Stage-A v3."""

from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn

from frank_eq.data.real_panel import ENTITY_NAMES, FrozenOperation
from frank_eq.models.layers import DotProductOperationDecoder
from frank_eq.utils import parameter_count

from .compiler import TokenSlotCompiler, active_coordinate_indices


def deterministic_token_features(token_ids: torch.Tensor, width: int) -> torch.Tensor:
    """Map public token IDs to fixed dense features without model activations."""

    if token_ids.ndim != 2 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a long tensor with shape [batch,tokens]")
    if width < 1:
        raise ValueError("token feature width must be positive")
    dimensions = torch.arange(width, device=token_ids.device, dtype=torch.float32)
    frequencies = 0.0001 + (dimensions.remainder(257.0) + 1.0) / 257.0
    phases = token_ids.to(torch.float32).unsqueeze(-1) * frequencies.view(1, 1, -1)
    feature = torch.sin(phases) + 0.5 * torch.cos(phases * 0.61803398875)
    return feature * (2.0 / 3.0)


class TokenIDResampler(nn.Module):
    """Parameter-matched token/position control with no source activations."""

    def __init__(self, **compiler_kwargs: int | float) -> None:
        super().__init__()
        self.compiler = TokenSlotCompiler(**compiler_kwargs)

    @property
    def input_width(self) -> int:
        return self.compiler.input_width

    @property
    def n_depths(self) -> int:
        return self.compiler.n_depths

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        entity_count: int,
    ) -> torch.Tensor:
        features = deterministic_token_features(token_ids, self.input_width)
        trajectory = features.unsqueeze(1).expand(-1, self.n_depths, -1, -1)
        return self.compiler(trajectory, attention_mask, entity_count=entity_count)


class FinalTokenPublicMLP(nn.Module):
    """Historical final-token public readout matched to a compiler parameter budget."""

    def __init__(
        self,
        *,
        input_width: int,
        n_depths: int,
        target_parameter_count: int,
        max_entities: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = input_width * n_depths
        output_dim = max_entities * (max_entities - 1)
        fixed_parameters = 2 * input_dim + output_dim
        per_hidden = input_dim + 1 + output_dim
        hidden_dim = max(1, round((target_parameter_count - fixed_parameters) / per_hidden))
        self.input_width = input_width
        self.n_depths = n_depths
        self.max_entities = max_entities
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        observed = parameter_count(self)
        relative_error = abs(observed - target_parameter_count) / target_parameter_count
        if relative_error > 0.05:
            raise RuntimeError(
                "final-token baseline cannot satisfy the frozen 5% parameter tolerance"
            )
        self.parameter_match = {
            "target": target_parameter_count,
            "observed": observed,
            "relative_error": relative_error,
            "hidden_dim": hidden_dim,
        }

    def forward(
        self,
        residuals: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        entity_count: int,
    ) -> torch.Tensor:
        if residuals.ndim != 4:
            raise ValueError("residuals must have shape [batch,depths,tokens,width]")
        if residuals.shape[1] != self.n_depths or residuals.shape[3] != self.input_width:
            raise ValueError("residuals disagree with final-token baseline dimensions")
        if attention_mask.shape != (residuals.shape[0], residuals.shape[2]):
            raise ValueError("attention mask has the wrong shape")
        final_indices = attention_mask.sum(dim=1).to(torch.long) - 1
        if torch.any(final_indices < 0):
            raise ValueError("every final-token example requires one unpadded token")
        gather_index = final_indices.view(-1, 1, 1, 1).expand(
            -1, self.n_depths, 1, self.input_width
        )
        final_tokens = residuals.gather(2, gather_index).squeeze(2).reshape(
            residuals.shape[0], -1
        )
        logits = self.network(final_tokens)
        indices = torch.tensor(
            active_coordinate_indices(entity_count, max_entities=self.max_entities),
            dtype=torch.long,
            device=residuals.device,
        )
        return logits.index_select(1, indices)


def unified_operation_descriptor(
    operation: FrozenOperation,
    *,
    entity_count: int,
    max_entities: int = 6,
) -> np.ndarray:
    """Lift a graph operation into one fixed six-entity learned-head descriptor."""

    families = (
        "lookup",
        "inverse",
        "mutual",
        "compose",
        "compare_outdegree",
        "counterfactual_add",
        "density",
        "reciprocity",
    )
    if not 2 <= entity_count <= max_entities:
        raise ValueError("entity count is outside the unified descriptor registry")
    descriptor = np.zeros(len(families) + 4 * max_entities + 1 + 2, dtype=np.float32)
    descriptor[families.index(operation.definition.family)] = 1.0
    offset = len(families)
    arguments = (
        operation.definition.fact_args[0],
        operation.definition.fact_args[1],
        operation.definition.residual_args[0],
        operation.definition.residual_args[1],
    )
    for block, argument in enumerate(arguments):
        descriptor[offset + block * max_entities + int(argument)] = 1.0
    descriptor[-3] = float(operation.definition.polarity)
    descriptor[-2 if entity_count == 4 else -1] = 1.0
    return descriptor


class HistoricalContinuousQuotient(nn.Module):
    """The registered private continuous-code plus learned-operation baseline."""

    def __init__(
        self,
        *,
        input_width: int,
        n_depths: int = 4,
        code_dim: int = 32,
        chart_hidden_dim: int = 160,
        operation_hidden_dim: int = 96,
        operation_descriptor_dim: int = 35,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.n_depths = n_depths
        input_dim = input_width * n_depths
        self.chart = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, chart_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chart_hidden_dim, chart_hidden_dim),
            nn.GELU(),
            nn.Linear(chart_hidden_dim, code_dim),
        )
        self.operation_head = DotProductOperationDecoder(
            code_dim=code_dim,
            operation_descriptor_dim=operation_descriptor_dim,
            hidden_dim=operation_hidden_dim,
            dropout=dropout,
        )

    def encode(
        self,
        residuals: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if residuals.ndim != 4:
            raise ValueError("continuous baseline expects [batch,depths,tokens,width]")
        final_indices = attention_mask.sum(dim=1).to(torch.long) - 1
        gather_index = final_indices.view(-1, 1, 1, 1).expand(
            -1, self.n_depths, 1, self.input_width
        )
        final_tokens = residuals.gather(2, gather_index).squeeze(2)
        return self.chart(final_tokens.reshape(final_tokens.shape[0], -1))

    def forward(
        self,
        residuals: torch.Tensor,
        attention_mask: torch.Tensor,
        operation_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encode(residuals, attention_mask)
        return self.operation_head(code, operation_descriptors), code


def parse_v3_world_prefix(text: str, entity_count: int) -> np.ndarray:
    """Recover the complete typed edge vector from any frozen v3 renderer."""

    names = ENTITY_NAMES[:entity_count]
    name_to_index = {name: index for index, name in enumerate(names)}
    edge = np.zeros((entity_count, entity_count), dtype=np.int8)

    canonical_matches = re.findall(r"^([A-Za-z]+)->([A-Za-z]+)=([01])$", text, re.MULTILINE)
    if canonical_matches:
        expected = entity_count * (entity_count - 1)
        if len(canonical_matches) != expected:
            raise ValueError("canonical edge-list renderer is incomplete")
        seen: set[tuple[int, int]] = set()
        for source_name, target_name, value in canonical_matches:
            if source_name not in name_to_index or target_name not in name_to_index:
                raise ValueError("canonical edge-list contains an unknown entity")
            coordinate = (name_to_index[source_name], name_to_index[target_name])
            if coordinate[0] == coordinate[1] or coordinate in seen:
                raise ValueError("canonical edge-list contains a duplicate or diagonal coordinate")
            seen.add(coordinate)
            edge[coordinate] = int(value)
        return edge[~np.eye(entity_count, dtype=bool)].astype(np.float32)

    adjacency_lines = {
        match.group(1): match.group(2)
        for match in re.finditer(r"^([A-Za-z]+) -> (.+)$", text, re.MULTILINE)
        if match.group(1) in name_to_index
    }
    if adjacency_lines:
        if set(adjacency_lines) != set(names):
            raise ValueError("adjacency renderer is incomplete")
        for source_name, targets in adjacency_lines.items():
            if targets == "none":
                continue
            for target_name in (value.strip() for value in targets.split(",")):
                if target_name not in name_to_index:
                    raise ValueError("adjacency renderer contains an unknown target")
                edge[name_to_index[source_name], name_to_index[target_name]] = 1
        return edge[~np.eye(entity_count, dtype=bool)].astype(np.float32)

    if not all(name in text for name in names):
        raise ValueError("natural renderer does not name every registered entity")
    for source_name, target_name in re.findall(
        r"^([A-Za-z]+) points to ([A-Za-z]+)\.$", text, re.MULTILINE
    ):
        if source_name not in name_to_index or target_name not in name_to_index:
            raise ValueError("natural renderer contains an unknown entity")
        edge[name_to_index[source_name], name_to_index[target_name]] = 1
    return edge[~np.eye(entity_count, dtype=bool)].astype(np.float32)
