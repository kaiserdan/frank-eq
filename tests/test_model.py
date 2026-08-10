import torch

from frank_eq.config import DataConfig, ModelConfig
from frank_eq.data.synthetic import generate_synthetic_bundle
from frank_eq.models import OperationalQuotientModel


def test_model_outputs_gauge_fixed_public_code() -> None:
    data_config = DataConfig(
        n_worlds=30,
        n_founder_models=2,
        include_held_model=True,
        n_renderers=2,
        n_layers=3,
        model_hidden_dims=[16, 20, 18],
        n_facts=6,
        n_residual=2,
        n_operations=24,
    )
    bundle = generate_synthetic_bundle(data_config)
    model = OperationalQuotientModel(
        model_hidden_dims=bundle.model_hidden_dims,
        n_layers=bundle.n_layers,
        n_facts=data_config.n_facts,
        n_residual=data_config.n_residual,
        operation_descriptor_dim=bundle.operation_descriptors.shape[1],
        config=ModelConfig(code_dim=12, chart_hidden_dim=32, operation_hidden_dim=24),
    )
    hidden = torch.from_numpy(bundle.hidden[:9]).float()
    model_ids = torch.from_numpy(bundle.model_ids[:9]).long()
    descriptors = torch.from_numpy(bundle.operation_descriptors).float()
    output = model(hidden, model_ids, descriptors)
    assert output.code.shape == (9, data_config.n_facts + data_config.n_residual)
    assert output.private_code.shape == (9, 12)
    assert output.signature_logits.shape == (9, data_config.n_operations)
    assert torch.all(output.code <= 1.0)
    assert torch.all(output.code >= -1.0)


def test_public_code_executes_without_chart() -> None:
    data_config = DataConfig(
        n_worlds=30,
        n_founder_models=2,
        include_held_model=False,
        n_renderers=2,
        model_hidden_dims=[16, 20],
        n_facts=6,
        n_residual=2,
        n_operations=24,
    )
    bundle = generate_synthetic_bundle(data_config)
    model = OperationalQuotientModel(
        model_hidden_dims=bundle.model_hidden_dims,
        n_layers=bundle.n_layers,
        n_facts=data_config.n_facts,
        n_residual=data_config.n_residual,
        operation_descriptor_dim=bundle.operation_descriptors.shape[1],
        config=ModelConfig(code_dim=12, chart_hidden_dim=32, operation_hidden_dim=24),
    )
    code = torch.zeros(4, data_config.n_facts + data_config.n_residual)
    descriptors = torch.from_numpy(bundle.operation_descriptors).float()
    logits, fact_logits, residual = model.decode_from_code(code, descriptors)
    assert logits.shape == (4, data_config.n_operations)
    assert fact_logits.shape == (4, data_config.n_facts)
    assert residual.shape == (4, data_config.n_residual)


def test_local_public_heads_onboard_complete_held_compiler() -> None:
    data_config = DataConfig(
        n_worlds=30,
        n_founder_models=2,
        include_held_model=True,
        n_renderers=2,
        model_hidden_dims=[16, 20, 18],
        n_facts=6,
        n_residual=2,
        n_operations=24,
    )
    bundle = generate_synthetic_bundle(data_config)
    model = OperationalQuotientModel(
        model_hidden_dims=bundle.model_hidden_dims,
        n_layers=bundle.n_layers,
        n_facts=data_config.n_facts,
        n_residual=data_config.n_residual,
        operation_descriptor_dim=bundle.operation_descriptors.shape[1],
        config=ModelConfig(
            code_dim=12,
            chart_hidden_dim=32,
            operation_hidden_dim=24,
            public_head_scope="local",
        ),
    )
    held_id = 2
    model.freeze_except_compiler(held_id)
    assert all(parameter.requires_grad for parameter in model.charts[str(held_id)].parameters())
    assert model.fact_heads is not None
    assert model.residual_heads is not None
    assert all(
        parameter.requires_grad for parameter in model.fact_heads[str(held_id)].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.residual_heads[str(held_id)].parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.decoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.charts["0"].parameters())

    hidden = torch.from_numpy(bundle.hidden[:12]).float()
    model_ids = torch.from_numpy(bundle.model_ids[:12]).long()
    descriptors = torch.from_numpy(bundle.operation_descriptors).float()
    output = model(hidden, model_ids, descriptors)
    assert output.fact_logits.shape == (12, data_config.n_facts)
    assert output.residual.shape == (12, data_config.n_residual)
