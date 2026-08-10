import torch
from torch.utils.data import DataLoader

from frank_eq.config import DataConfig, LossConfig, ModelConfig
from frank_eq.data.synthetic import ObservationDataset, WorldBatchSampler, generate_synthetic_bundle
from frank_eq.models import OperationalQuotientModel
from frank_eq.training.objectives import compute_objective


def test_world_sampler_keeps_complete_view_groups() -> None:
    config = DataConfig(
        n_worlds=30,
        n_founder_models=2,
        include_held_model=False,
        n_renderers=2,
        model_hidden_dims=[16, 20],
        n_facts=6,
        n_residual=2,
        n_operations=24,
    )
    bundle = generate_synthetic_bundle(config)
    indices = bundle.indices_for(
        world_ids=bundle.split.train_world_ids,
        model_ids=bundle.split.founder_model_ids,
    )
    dataset = ObservationDataset(bundle, indices)
    sampler = WorldBatchSampler(dataset, worlds_per_batch=2, shuffle=False, seed=1)
    first = next(iter(sampler))
    worlds = {int(dataset[position]["world_id"]) for position in first}
    assert len(worlds) == 2
    expected_views = len(worlds) * config.n_founder_models * config.n_renderers
    assert len(first) == expected_views


def test_objective_is_finite_and_backpropagates() -> None:
    config = DataConfig(
        n_worlds=30,
        n_founder_models=2,
        include_held_model=False,
        n_renderers=2,
        model_hidden_dims=[16, 20],
        n_facts=6,
        n_residual=2,
        n_operations=24,
    )
    bundle = generate_synthetic_bundle(config)
    indices = bundle.indices_for(
        world_ids=bundle.split.train_world_ids[:2],
        model_ids=bundle.split.founder_model_ids,
    )
    dataset = ObservationDataset(bundle, indices)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    model_config = ModelConfig(code_dim=12, chart_hidden_dim=32, operation_hidden_dim=24)
    model = OperationalQuotientModel(
        model_hidden_dims=bundle.model_hidden_dims,
        n_layers=bundle.n_layers,
        n_facts=config.n_facts,
        n_residual=config.n_residual,
        operation_descriptor_dim=bundle.operation_descriptors.shape[1],
        config=model_config,
    )
    descriptors = torch.from_numpy(bundle.operation_descriptors).float()
    output = model(batch["hidden"], batch["model_id"], descriptors)
    losses = compute_objective(
        model=model,
        output=output,
        batch=batch,
        train_operation_ids=torch.tensor(bundle.split.train_operation_ids),
        loss_config=LossConfig(),
        model_config=model_config,
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
