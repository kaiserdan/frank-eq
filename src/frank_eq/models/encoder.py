"""Future-defined operational quotient model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from frank_eq.config import ModelConfig

from .layers import (
    GradientReversal,
    GraphOperationDecoder,
    PublicOperationDecoder,
    StraightThroughQuantizer,
    WorkspaceGate,
)


@dataclass(slots=True)
class QuotientOutput:
    code: torch.Tensor
    private_code: torch.Tensor
    signature_logits: torch.Tensor
    fact_logits: torch.Tensor
    residual: torch.Tensor
    model_logits: torch.Tensor


def _make_public_head(
    code_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(code_dim),
        nn.Linear(code_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ModelLocalChart(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, config: ModelConfig):
        super().__init__()
        input_dim = n_layers * hidden_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.workspace_gate = WorkspaceGate(input_dim, config.use_workspace_gate)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.chart_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.chart_hidden_dim, config.chart_hidden_dim),
            nn.GELU(),
            nn.Linear(config.chart_hidden_dim, config.code_dim),
        )
        self.quantizer = StraightThroughQuantizer(config.quantization_bits)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"expected hidden [batch,layers,width], got {tuple(hidden.shape)}")
        if hidden.shape[1] != self.n_layers or hidden.shape[2] < self.hidden_dim:
            raise ValueError("hidden shape is incompatible with chart")
        flattened = hidden[:, :, : self.hidden_dim].reshape(hidden.shape[0], -1)
        gated = self.workspace_gate(flattened)
        return self.quantizer(self.network(gated))


class OperationalQuotientModel(nn.Module):
    """Model-local compilers coupled through a frozen public future decoder.

    ``public_head_scope=shared`` preserves the original Stage-A v1 and
    synthetic checkpoint contract. ``public_head_scope=local`` gives every
    sender a complete model-local compiler (chart plus fact/residual heads)
    while keeping only the public state coordinates and interrogator shared.
    """

    def __init__(
        self,
        *,
        model_hidden_dims: list[int],
        n_layers: int,
        n_facts: int,
        n_residual: int,
        operation_descriptor_dim: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.model_hidden_dims = list(model_hidden_dims)
        self.n_layers = n_layers
        self.n_facts = n_facts
        self.n_residual = n_residual
        self.config = config
        self.public_head_scope = config.public_head_scope
        self.charts = nn.ModuleDict(
            {
                str(model_id): ModelLocalChart(n_layers, hidden_dim, config)
                for model_id, hidden_dim in enumerate(model_hidden_dims)
            }
        )
        if config.decoder_type == "public_coefficients":
            self.decoder = PublicOperationDecoder(
                n_facts=n_facts,
                n_residual=n_residual,
                descriptor_dim=operation_descriptor_dim,
            )
        elif config.decoder_type == "graph":
            self.decoder = GraphOperationDecoder(
                n_entities=config.graph_n_entities,
                n_residual=n_residual,
                descriptor_dim=operation_descriptor_dim,
                temperature=config.graph_temperature,
            )
        else:
            raise ValueError(f"unsupported decoder_type: {config.decoder_type}")

        public_hidden = max(1, config.chart_hidden_dim // 2)
        if self.public_head_scope == "shared":
            self.fact_head = _make_public_head(config.code_dim, public_hidden, n_facts)
            self.residual_head = _make_public_head(config.code_dim, public_hidden, n_residual)
            self.fact_heads = None
            self.residual_heads = None
        elif self.public_head_scope == "local":
            self.fact_head = None
            self.residual_head = None
            self.fact_heads = nn.ModuleDict(
                {
                    str(model_id): _make_public_head(config.code_dim, public_hidden, n_facts)
                    for model_id in range(len(model_hidden_dims))
                }
            )
            self.residual_heads = nn.ModuleDict(
                {
                    str(model_id): _make_public_head(config.code_dim, public_hidden, n_residual)
                    for model_id in range(len(model_hidden_dims))
                }
            )
        else:
            raise ValueError(
                "public_head_scope must be 'shared' or 'local', "
                f"got {self.public_head_scope!r}"
            )

        self.gradient_reversal = GradientReversal(config.gradient_reversal_weight)
        public_dim = n_facts + n_residual
        self.model_classifier = nn.Sequential(
            nn.Linear(public_dim, max(16, public_dim)),
            nn.GELU(),
            nn.Linear(max(16, public_dim), len(model_hidden_dims)),
        )
        self.residual_public_scale = 2.0

    def encode_private(self, hidden: torch.Tensor, model_ids: torch.Tensor) -> torch.Tensor:
        if hidden.shape[0] != model_ids.shape[0]:
            raise ValueError("hidden and model_ids batch dimensions differ")
        output = torch.empty(
            hidden.shape[0],
            self.config.code_dim,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        for model_id in torch.unique(model_ids).tolist():
            model_id = int(model_id)
            if str(model_id) not in self.charts:
                raise KeyError(f"no chart registered for model {model_id}")
            selection = model_ids == model_id
            output[selection] = self.charts[str(model_id)](hidden[selection])
        return output

    def _decode_public_coordinates(
        self,
        private_code: torch.Tensor,
        model_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.public_head_scope == "shared":
            if self.fact_head is None or self.residual_head is None:
                raise RuntimeError("shared public heads are not initialized")
            return self.fact_head(private_code), self.residual_head(private_code)

        if self.fact_heads is None or self.residual_heads is None:
            raise RuntimeError("local public heads are not initialized")
        fact_logits = private_code.new_empty((len(private_code), self.n_facts))
        residual = private_code.new_empty((len(private_code), self.n_residual))
        for model_id in torch.unique(model_ids).tolist():
            model_id = int(model_id)
            selection = model_ids == model_id
            fact_logits[selection] = self.fact_heads[str(model_id)](private_code[selection])
            residual[selection] = self.residual_heads[str(model_id)](private_code[selection])
        return fact_logits, residual

    def build_public_code(
        self,
        fact_logits: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        signed_facts = 2.0 * torch.sigmoid(fact_logits) - 1.0
        bounded_residual = torch.clamp(
            residual / self.residual_public_scale,
            -1.0,
            1.0,
        )
        return torch.cat([signed_facts, bounded_residual], dim=1)

    def decode_from_code(
        self,
        code: torch.Tensor,
        operation_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = self.n_facts + self.n_residual
        if code.ndim != 2 or code.shape[1] != expected:
            raise ValueError(f"expected public code [batch,{expected}], got {tuple(code.shape)}")
        fact_probabilities = torch.clamp(
            (code[:, : self.n_facts] + 1.0) * 0.5,
            1e-5,
            1.0 - 1e-5,
        )
        fact_logits = torch.logit(fact_probabilities)
        residual = code[:, self.n_facts :] * self.residual_public_scale
        signature_logits = self.decoder(fact_logits, residual, operation_descriptors)
        return signature_logits, fact_logits, residual

    def forward(
        self,
        hidden: torch.Tensor,
        model_ids: torch.Tensor,
        operation_descriptors: torch.Tensor,
    ) -> QuotientOutput:
        private_code = self.encode_private(hidden, model_ids)
        fact_logits, residual = self._decode_public_coordinates(private_code, model_ids)
        code = self.build_public_code(fact_logits, residual)
        signature_logits = self.decoder(fact_logits, residual, operation_descriptors)
        model_logits = self.model_classifier(self.gradient_reversal(code))
        return QuotientOutput(
            code=code,
            private_code=private_code,
            signature_logits=signature_logits,
            fact_logits=fact_logits,
            residual=residual,
            model_logits=model_logits,
        )

    def workspace_regularization(self, model_ids: torch.Tensor) -> torch.Tensor:
        unique_ids = sorted(int(value) for value in torch.unique(model_ids).tolist())
        if not unique_ids:
            return torch.tensor(0.0, device=model_ids.device)
        return torch.stack(
            [self.charts[str(model_id)].workspace_gate.regularization() for model_id in unique_ids]
        ).mean()

    def hard_quantize_code(self, code: torch.Tensor, bits: int) -> torch.Tensor:
        first_chart = next(iter(self.charts.values()))
        return first_chart.quantizer.hard_quantize(code, bits=bits)

    def freeze_except_compiler(self, model_id: int) -> None:
        """Freeze every component except one sender's complete local compiler."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.charts[str(model_id)].parameters():
            parameter.requires_grad = True
        if self.public_head_scope == "local":
            if self.fact_heads is None or self.residual_heads is None:
                raise RuntimeError("local public heads are not initialized")
            for parameter in self.fact_heads[str(model_id)].parameters():
                parameter.requires_grad = True
            for parameter in self.residual_heads[str(model_id)].parameters():
                parameter.requires_grad = True

    def freeze_except_chart(self, model_id: int) -> None:
        """Backward-compatible alias for the v1 held-sender behavior."""

        self.freeze_except_compiler(model_id)

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
