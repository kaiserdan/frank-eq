from frank_eq.real_config import RealModelSpec, RealRunConfig


def test_real_config_builds_existing_stage0_contract() -> None:
    config = RealRunConfig(
        models=[
            RealModelSpec("a", "org/a", "founder"),
            RealModelSpec("b", "org/b", "founder"),
            RealModelSpec("c", "org/c", "held"),
        ]
    )
    config.validate()
    stage0 = config.to_stage0_config([128, 160, 192])
    assert stage0.model.decoder_type == "graph"
    assert stage0.data.n_models == 3
    assert stage0.data.n_facts == config.panel.n_entities * (config.panel.n_entities - 1)
