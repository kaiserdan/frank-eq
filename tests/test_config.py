from pathlib import Path

import pytest

from frank_eq.config import RunConfig, load_config


def test_default_config_is_valid() -> None:
    config = RunConfig()
    config.validate()
    assert config.data.n_models == 4


def test_partial_yaml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("run_name: partial\ndata:\n  n_worlds: 48\n")
    config = load_config(path)
    assert config.run_name == "partial"
    assert config.output_dir == "runs/stage0"
    assert config.data.n_worlds == 48


def test_unknown_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_config(path)
