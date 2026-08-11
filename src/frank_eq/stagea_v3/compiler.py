"""Query-blind token/slot compilers for the Stage-A v3 public edge basis."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


def canonical_coordinates(max_entities: int = 6) -> tuple[tuple[int, int], ...]:
    """Return the public row-major non-diagonal coordinate registry."""

    if max_entities < 2:
        raise ValueError("max_entities must be at least two")
    return tuple(
        (source, target)
        for source in range(max_entities)
        for target in range(max_entities)
        if source != target
    )


def active_coordinate_indices(
    entity_count: int,
    *,
    max_entities: int = 6,
) -> tuple[int, ...]:
    """Select an entity-count subgraph without changing public coordinate meaning."""

    if not 2 <= entity_count <= max_entities:
        raise ValueError(
            f"entity_count must be between 2 and {max_entities}; got {entity_count}"
        )
    return tuple(
        index
        for index, (source, target) in enumerate(canonical_coordinates(max_entities))
        if source < entity_count and target < entity_count
    )


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        attention_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_slots = self.query_norm(slots)
        attended, _ = self.cross_attention(
            normalized_slots,
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        slots = slots + self.attention_dropout(attended)
        return slots + self.feedforward(self.feedforward_norm(slots))


class TokenSlotCompiler(nn.Module):
    """Compile all-token residual trajectories into typed directed-edge logits.

    The module has deliberately no operation or label argument. Each instance is
    local to one source model and one target channel.
    """

    def __init__(
        self,
        *,
        input_width: int,
        n_depths: int = 4,
        max_entities: int = 6,
        max_tokens: int = 512,
        model_dim: int = 192,
        attention_heads: int = 6,
        attention_blocks: int = 2,
        feedforward_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_width < 1 or n_depths < 1 or max_tokens < 1:
            raise ValueError("input_width, n_depths, and max_tokens must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if attention_blocks < 1:
            raise ValueError("attention_blocks must be positive")
        self.input_width = input_width
        self.n_depths = n_depths
        self.max_entities = max_entities
        self.max_tokens = max_tokens
        self.model_dim = model_dim
        self.coordinates = canonical_coordinates(max_entities)

        self.depth_projections = nn.ModuleList(
            nn.Linear(input_width, model_dim) for _ in range(n_depths)
        )
        self.depth_embeddings = nn.Parameter(torch.empty(n_depths, model_dim))
        self.position_embedding = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim),
        )
        self.coordinate_queries = nn.Parameter(torch.empty(len(self.coordinates), model_dim))
        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(
                model_dim=model_dim,
                attention_heads=attention_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(attention_blocks)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.coordinate_head_weight = nn.Parameter(
            torch.empty(len(self.coordinates), model_dim)
        )
        self.coordinate_head_bias = nn.Parameter(torch.zeros(len(self.coordinates)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.depth_embeddings, std=0.02)
        nn.init.normal_(self.coordinate_queries, std=0.02)
        nn.init.normal_(self.coordinate_head_weight, std=self.model_dim**-0.5)

    def _validate_inputs(
        self,
        residuals: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_count: int,
    ) -> None:
        if residuals.ndim != 4:
            raise ValueError("residuals must have shape [batch, depths, tokens, width]")
        batch, depths, tokens, width = residuals.shape
        if depths != self.n_depths or width != self.input_width:
            raise ValueError(
                "residual trajectory shape does not match compiler depths/input width"
            )
        if tokens > self.max_tokens:
            raise ValueError(
                f"residual trajectory has {tokens} tokens; compiler maximum is {self.max_tokens}"
            )
        if attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape [batch, tokens]")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean")
        if torch.any(attention_mask.sum(dim=1) < 1):
            raise ValueError("every compiler example must contain at least one unpadded token")
        active_coordinate_indices(entity_count, max_entities=self.max_entities)

    def forward(
        self,
        residuals: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        entity_count: int,
    ) -> torch.Tensor:
        self._validate_inputs(residuals, attention_mask, entity_count)
        batch, _, tokens, _ = residuals.shape

        if tokens == 1:
            normalized_positions = torch.zeros(
                (1, 1, 1), dtype=residuals.dtype, device=residuals.device
            )
        else:
            normalized_positions = torch.linspace(
                0.0,
                1.0,
                tokens,
                dtype=residuals.dtype,
                device=residuals.device,
            ).view(1, tokens, 1)
        position_embeddings = self.position_embedding(normalized_positions)

        projected_depths = []
        for depth, projection in enumerate(self.depth_projections):
            projected = projection(residuals[:, depth])
            projected = projected + self.depth_embeddings[depth].view(1, 1, -1)
            projected_depths.append(projected + position_embeddings)
        memory = torch.stack(projected_depths, dim=1).reshape(
            batch, self.n_depths * tokens, self.model_dim
        )
        memory_padding_mask = (~attention_mask).unsqueeze(1).expand(
            -1, self.n_depths, -1
        ).reshape(batch, self.n_depths * tokens)

        indices = torch.tensor(
            active_coordinate_indices(entity_count, max_entities=self.max_entities),
            dtype=torch.long,
            device=residuals.device,
        )
        slots = self.coordinate_queries.index_select(0, indices).unsqueeze(0).expand(
            batch, -1, -1
        )
        for block in self.blocks:
            slots = block(slots, memory, memory_padding_mask=memory_padding_mask)
        slots = self.output_norm(slots)
        weights = self.coordinate_head_weight.index_select(0, indices)
        biases = self.coordinate_head_bias.index_select(0, indices)
        return torch.einsum("bcd,cd->bc", slots, weights) + biases


@contextmanager
def _forked_seed(seed: int) -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


class IndependentChannelCompilers(nn.Module):
    """Own semantic and behavioral compilers with provably disjoint parameters."""

    def __init__(self, *, seed: int, **compiler_kwargs: int | float) -> None:
        super().__init__()
        with _forked_seed(seed * 2 + 1):
            self.semantic = TokenSlotCompiler(**compiler_kwargs)
        with _forked_seed(seed * 2 + 2):
            self.behavioral = TokenSlotCompiler(**compiler_kwargs)
        self.assert_parameter_disjointness()

    def assert_parameter_disjointness(self) -> None:
        semantic_ids = {id(parameter) for parameter in self.semantic.parameters()}
        behavioral_ids = {id(parameter) for parameter in self.behavioral.parameters()}
        if semantic_ids & behavioral_ids:
            raise RuntimeError("semantic and behavioral compilers share trainable parameters")

    def channel(self, name: str) -> TokenSlotCompiler:
        if name == "semantic":
            return self.semantic
        if name == "behavioral":
            return self.behavioral
        raise ValueError(f"unsupported Stage-A v3 channel: {name!r}")

    def forward(
        self,
        channel: str,
        residuals: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        entity_count: int,
    ) -> torch.Tensor:
        return self.channel(channel)(
            residuals,
            attention_mask,
            entity_count=entity_count,
        )
