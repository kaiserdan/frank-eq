"""Fail-closed process and test-access controls for Stage-A v3."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from frank_eq.utils import atomic_write_json, sha256_file

_STAGES = ("prepare", "founder_fit", "freeze", "held_onboard", "evaluate")
_MANIFESTS = {
    "freeze": ("freeze_manifest.json", "frank_eq_stagea_v3_freeze_v1"),
    "held_onboard": (
        "held_onboarding_manifest.json",
        "frank_eq_stagea_v3_held_onboarding_v1",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


class StageAV3AccessController:
    """Serialize stage transitions and consume the one registered test access."""

    def __init__(self, run_root: str | Path, *, config_sha256: str) -> None:
        self.run_root = Path(run_root).resolve()
        self.config_sha256 = config_sha256
        self.ledger_path = self.run_root / "access_ledger.json"
        self.lock_path = self.run_root / ".stagea_v3_access.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _initial_ledger(self) -> dict[str, Any]:
        timestamp = _utc_now()
        return {
            "schema": "frank_eq_stagea_v3_access_ledger_v1",
            "config_sha256": self.config_sha256,
            "current_stage": "prepare",
            "test_access_limit": 1,
            "test_access_count": 0,
            "registered_test_files": [],
            "test_file_opens": [],
            "events": [
                {
                    "event": "stage_entered",
                    "stage": "prepare",
                    "timestamp_utc": timestamp,
                }
            ],
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.ledger_path.is_file():
            ledger = self._initial_ledger()
            atomic_write_json(self.ledger_path, ledger)
            return ledger
        payload = json.loads(self.ledger_path.read_text())
        if payload.get("schema") != "frank_eq_stagea_v3_access_ledger_v1":
            raise ValueError("unsupported Stage-A v3 access-ledger schema")
        if payload.get("config_sha256") != self.config_sha256:
            raise ValueError("access ledger belongs to a different frozen config")
        if payload.get("current_stage") not in _STAGES:
            raise ValueError("access ledger contains an invalid process stage")
        if payload.get("test_access_limit") != 1:
            raise ValueError("access ledger changed the frozen one-access limit")
        count = payload.get("test_access_count")
        if not isinstance(count, int) or not 0 <= count <= 1:
            raise ValueError("access ledger contains an invalid test-access count")
        return payload

    def initialize(self) -> dict[str, Any]:
        with self._locked():
            return self._read_unlocked()

    def read(self) -> dict[str, Any]:
        with self._locked():
            return self._read_unlocked()

    def _resolve_artifact(self, relative_path: str) -> Path:
        candidate = (self.run_root / relative_path).resolve()
        if candidate != self.run_root and self.run_root not in candidate.parents:
            raise ValueError("artifact path escapes the Stage-A v3 run root")
        return candidate

    def _validate_manifest_unlocked(self, stage: str) -> dict[str, Any]:
        filename, schema = _MANIFESTS[stage]
        path = self.run_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"required Stage-A v3 manifest does not exist: {filename}")
        payload = json.loads(path.read_text())
        if payload.get("schema") != schema or payload.get("status") != "frozen":
            raise ValueError(f"{filename} is not a frozen {schema} manifest")
        if payload.get("config_sha256") != self.config_sha256:
            raise ValueError(f"{filename} belongs to a different frozen config")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"{filename} must bind at least one artifact")
        for relative_path, expected_sha256 in artifacts.items():
            artifact = self._resolve_artifact(str(relative_path))
            if not artifact.is_file() or sha256_file(artifact) != expected_sha256:
                raise ValueError(f"{filename} artifact hash mismatch: {relative_path}")
        return payload

    def advance(self, target_stage: str) -> dict[str, Any]:
        if target_stage not in _STAGES:
            raise ValueError(f"unsupported Stage-A v3 stage: {target_stage}")
        with self._locked():
            ledger = self._read_unlocked()
            current_index = _STAGES.index(str(ledger["current_stage"]))
            target_index = _STAGES.index(target_stage)
            if target_index != current_index + 1:
                raise RuntimeError(
                    f"Stage-A v3 transition must be sequential; "
                    f"current={ledger['current_stage']}, requested={target_stage}"
                )
            if target_stage in {"freeze", "held_onboard"}:
                self._validate_manifest_unlocked("freeze")
            elif target_stage == "evaluate":
                raise RuntimeError("evaluate is entered only by consuming the test access")
            ledger["current_stage"] = target_stage
            ledger["events"].append(
                {
                    "event": "stage_entered",
                    "stage": target_stage,
                    "timestamp_utc": _utc_now(),
                }
            )
            atomic_write_json(self.ledger_path, ledger)
            return ledger

    def assert_can_create_test(self, test_files: list[str]) -> dict[str, Any]:
        """Validate both freezes, atomically consume test access, and enter evaluation."""

        if not test_files or len(test_files) != len(set(test_files)):
            raise ValueError("test_files must be a non-empty unique list")
        with self._locked():
            ledger = self._read_unlocked()
            if ledger["current_stage"] != "held_onboard":
                raise RuntimeError("test creation is forbidden before held onboarding is frozen")
            if ledger["test_access_count"] != 0:
                raise RuntimeError("the frozen Stage-A v3 test access has already been consumed")
            self._validate_manifest_unlocked("freeze")
            self._validate_manifest_unlocked("held_onboard")
            normalized: list[str] = []
            for relative_path in test_files:
                candidate = self._resolve_artifact(relative_path)
                if candidate.exists():
                    raise RuntimeError(
                        f"test artifact exists before registered test creation: {relative_path}"
                    )
                normalized.append(str(candidate.relative_to(self.run_root)))
            timestamp = _utc_now()
            ledger["current_stage"] = "evaluate"
            ledger["test_access_count"] = 1
            ledger["registered_test_files"] = normalized
            ledger["events"].extend(
                [
                    {
                        "event": "test_access_consumed",
                        "stage": "evaluate",
                        "test_files": normalized,
                        "timestamp_utc": timestamp,
                    },
                    {
                        "event": "stage_entered",
                        "stage": "evaluate",
                        "timestamp_utc": timestamp,
                    },
                ]
            )
            atomic_write_json(self.ledger_path, ledger)
            return ledger

    def record_test_file_open(self, relative_path: str) -> dict[str, Any]:
        """Record every sanctioned opening of a previously registered test artifact."""

        with self._locked():
            ledger = self._read_unlocked()
            normalized = str(self._resolve_artifact(relative_path).relative_to(self.run_root))
            if ledger["current_stage"] != "evaluate" or normalized not in ledger[
                "registered_test_files"
            ]:
                raise RuntimeError("opening an unregistered Stage-A v3 test artifact is forbidden")
            artifact = self._resolve_artifact(normalized)
            if not artifact.is_file():
                raise FileNotFoundError(f"registered test artifact does not exist: {normalized}")
            ledger["test_file_opens"].append(
                {
                    "path": normalized,
                    "sha256": sha256_file(artifact),
                    "timestamp_utc": _utc_now(),
                }
            )
            atomic_write_json(self.ledger_path, ledger)
            return ledger
