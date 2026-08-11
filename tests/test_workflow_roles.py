import datetime as dt
from pathlib import Path

import pytest

from frank_eq.rate_compute.workflow import _timestamp as rate_compute_timestamp
from frank_eq.real_config import load_real_config
from frank_eq.workflow import _timestamp as real_workflow_timestamp
from frank_eq.workflow import infer_protocol_role, validate_real_stage_role

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("timestamp", [real_workflow_timestamp, rate_compute_timestamp])
def test_workflow_timestamp_uses_python310_compatible_utc(timestamp) -> None:
    parsed = dt.datetime.fromisoformat(timestamp())
    assert parsed.utcoffset() == dt.timedelta(0)


def test_stageq_role_allows_only_cache_and_validate() -> None:
    path = ROOT / "configs/stageq/real_lumi_chat_turn.yaml"
    config = load_real_config(path)
    assert infer_protocol_role(config, path) == "stageq"
    role, stages = validate_real_stage_role(config, path, "cache,validate")
    assert role == "stageq"
    assert stages == ("cache", "validate")
    with pytest.raises(ValueError, match="permit only cache,validate"):
        validate_real_stage_role(config, path, "cache,validate,train,eval")
    with pytest.raises(ValueError, match="permit only cache,validate"):
        validate_real_stage_role(config, path, "cache,validate,diagnose")


def test_stagea_role_keeps_historical_workflow_surface() -> None:
    path = ROOT / "configs/stage0/real_lumi_v2.yaml"
    config = load_real_config(path)
    assert infer_protocol_role(config, path) == "stagea"
    role, stages = validate_real_stage_role(config, path, "cache,validate,train,eval")
    assert role == "stagea"
    assert stages == ("cache", "validate", "train", "eval")
