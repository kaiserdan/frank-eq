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


def test_real_config_requires_revision_pins_when_flagged() -> None:
    config = RealRunConfig(
        require_revision_pins=True,
        models=[
            RealModelSpec("a", "org/a", "founder"),
            RealModelSpec("b", "org/b", "founder"),
            RealModelSpec("c", "org/c", "held"),
        ],
    )
    try:
        config.validate()
    except ValueError as error:
        assert "revision" in str(error)
    else:
        raise AssertionError("missing revision pins must fail validation")

    config.models[0].revision = "abc123"
    config.models[1].revision = "abc123"
    config.models[2].revision = "abc123"
    config.validate()


def test_real_config_capture_prompt_format_and_parity() -> None:
    config = RealRunConfig(
        models=[
            RealModelSpec("a", "org/a", "founder"),
            RealModelSpec("b", "org/b", "founder"),
            RealModelSpec("c", "org/c", "held"),
        ],
    )
    config.capture.prompt_format = "chat"
    config.capture.parity_sample_size = 8
    config.validate()

    config.capture.prompt_format = "template"
    try:
        config.validate()
    except ValueError as error:
        assert "prompt_format" in str(error)
    else:
        raise AssertionError("invalid prompt_format must fail validation")

    config.capture.prompt_format = "chat"
    config.capture.parity_sample_size = -1
    try:
        config.validate()
    except ValueError as error:
        assert "parity_sample_size" in str(error)
    else:
        raise AssertionError("negative parity sample must fail validation")
