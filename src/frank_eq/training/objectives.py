"""Objective functions for operation-agnostic state learning."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from frank_eq.config import LossConfig, ModelConfig
from frank_eq.models import OperationalQuotientModel, QuotientOutput


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    signature: torch.Tensor
    facts: torch.Tensor
    residual: torch.Tensor
    renderer_invariance: torch.Tensor
    cross_model_invariance: torch.Tensor
    world_contrastive: torch.Tensor
    model_adversary: torch.Tensor
    code_variance: torch.Tensor
    workspace: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            field.name: float(getattr(self, field.name).detach().cpu().item())
            for field in fields(self)
        }


def _group_mean_loss(
    code: torch.Tensor,
    primary_ids: torch.Tensor,
    secondary_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize dispersion around group means without assuming batch ordering."""

    if secondary_ids is None:
        group_keys = primary_ids
    else:
        multiplier = int(secondary_ids.max().item()) + 1 if secondary_ids.numel() else 1
        group_keys = primary_ids * multiplier + secondary_ids
    losses: list[torch.Tensor] = []
    for key in torch.unique(group_keys):
        selection = group_keys == key
        if int(selection.sum()) < 2:
            continue
        group = code[selection]
        losses.append(((group - group.mean(dim=0, keepdim=True)) ** 2).mean())
    if not losses:
        return code.new_tensor(0.0)
    return torch.stack(losses).mean()



def _world_contrastive_loss(
    code: torch.Tensor,
    world_ids: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    if code.shape[0] < 2:
        return code.new_tensor(0.0)
    normalized = F.normalize(code, dim=1)
    similarities = normalized @ normalized.transpose(0, 1) / temperature
    identity = torch.eye(code.shape[0], dtype=torch.bool, device=code.device)
    positive = world_ids[:, None].eq(world_ids[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not valid.any():
        return code.new_tensor(0.0)
    denominator = torch.logsumexp(similarities.masked_fill(identity, float("-inf")), dim=1)
    numerator = torch.logsumexp(similarities.masked_fill(~positive, float("-inf")), dim=1)
    return -(numerator[valid] - denominator[valid]).mean()

def compute_objective(
    *,
    model: OperationalQuotientModel,
    output: QuotientOutput,
    batch: dict[str, torch.Tensor],
    train_operation_ids: torch.Tensor,
    loss_config: LossConfig,
    model_config: ModelConfig,
    onboarding: bool = False,
) -> LossBreakdown:
    operation_selection = train_operation_ids.long()
    signature_loss = F.binary_cross_entropy_with_logits(
        output.signature_logits[:, operation_selection],
        batch["signatures"][:, operation_selection],
    )
    fact_loss = F.binary_cross_entropy_with_logits(output.fact_logits, batch["facts"])
    residual_loss = F.mse_loss(output.residual, batch["residual"])
    renderer_loss = _group_mean_loss(output.code, batch["world_id"], batch["model_id"])
    cross_model_loss = _group_mean_loss(output.code, batch["world_id"])
    world_contrastive_loss = _world_contrastive_loss(output.code, batch["world_id"])
    if onboarding:
        adversary_loss = output.code.new_tensor(0.0)
        cross_model_loss = output.code.new_tensor(0.0)
    else:
        adversary_loss = F.cross_entropy(output.model_logits, batch["model_id"])

    standard_deviation = output.code.std(dim=0, unbiased=False)
    variance_loss = torch.relu(loss_config.code_variance_floor - standard_deviation).pow(2).mean()
    workspace_loss = model.workspace_regularization(batch["model_id"])

    total = (
        loss_config.signature * signature_loss
        + loss_config.facts * fact_loss
        + loss_config.residual * residual_loss
        + loss_config.renderer_invariance * renderer_loss
        + loss_config.cross_model_invariance * cross_model_loss
        + loss_config.world_contrastive * world_contrastive_loss
        + loss_config.model_adversary * adversary_loss
        + variance_loss
        + model_config.workspace_l1 * workspace_loss
    )
    return LossBreakdown(
        total=total,
        signature=signature_loss,
        facts=fact_loss,
        residual=residual_loss,
        renderer_invariance=renderer_loss,
        cross_model_invariance=cross_model_loss,
        world_contrastive=world_contrastive_loss,
        model_adversary=adversary_loss,
        code_variance=variance_loss,
        workspace=workspace_loss,
    )
