import numpy as np
import torch

from frank_eq.data.real_panel import (
    build_operation_descriptor,
    evaluate_operation,
    generate_real_panel,
    render_operation_query,
    render_world_prefix,
)
from frank_eq.models.layers import GraphOperationDecoder
from frank_eq.real_config import GRAPH_OPERATION_FAMILIES, RealPanelConfig


def test_real_panel_is_balanced_and_query_blind() -> None:
    config = RealPanelConfig(n_worlds=32, n_entities=5, n_operations=16, seed=17)
    panel = generate_real_panel(config)
    labels = np.asarray(panel.oracle_signatures) >= 0.5
    assert labels.shape == (32, 16)
    assert labels.mean(axis=0).min() >= config.min_operation_positive_fraction
    assert labels.mean(axis=0).max() <= config.max_operation_positive_fraction
    prefix = render_world_prefix(panel.worlds[0], 0)
    query = render_operation_query(panel.operations[0].definition, 5, " A", " B")
    assert "Registered operation" not in prefix
    assert "Registered operation" in query
    assert render_world_prefix(panel.worlds[0], 0) != render_world_prefix(panel.worlds[0], 1)


def test_graph_decoder_matches_near_discrete_oracle() -> None:
    config = RealPanelConfig(n_worlds=32, n_entities=5, n_operations=16, seed=21)
    panel = generate_real_panel(config)
    descriptors = torch.tensor(
        np.stack([build_operation_descriptor(row.definition, 5) for row in panel.operations]),
        dtype=torch.float32,
    )
    decoder = GraphOperationDecoder(
        n_entities=5,
        n_residual=2,
        descriptor_dim=descriptors.shape[1],
        temperature=10.0,
    )
    world = panel.worlds[0]
    facts = torch.tensor(world.fact_vector()).unsqueeze(0)
    fact_logits = torch.where(facts > 0.5, torch.tensor(12.0), torch.tensor(-12.0))
    residual = torch.tensor(world.residual_vector()).unsqueeze(0)
    probabilities = torch.sigmoid(decoder(fact_logits, residual, descriptors))[0].detach().numpy()
    truth = np.asarray([evaluate_operation(world, row.definition) for row in panel.operations])
    assert np.mean((probabilities >= 0.5) == truth) >= 0.95
    assert len(GRAPH_OPERATION_FAMILIES) == 8
