import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frank_eq.cluster import ClusterClient, build_source_archive


def test_source_archive_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "src").mkdir()
    (root / "src" / "x.py").write_text("x = 1\n")
    first = build_source_archive(root, tmp_path / "archives")
    second = build_source_archive(root, tmp_path / "archives")
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == 2


def test_cluster_submission_dry_run_needs_no_network(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "frank_eq").mkdir(parents=True)
    (root / "configs" / "stage0").mkdir(parents=True)
    (root / "configs" / "stage0" / "real.yaml").write_text("run_name: test\n")
    (root / "olivia").mkdir()
    (root / "olivia" / "run.slurm").write_text("#!/bin/bash\n")
    client = ClusterClient("olivia", root=root)
    plan = client.submit(
        job_name="frank-eq-test",
        config_path="configs/stage0/real.yaml",
        profile="smoke",
        stages="cache,validate",
        dry_run=True,
    )
    assert plan["dry_run"] is True
    assert plan["source"]["file_count"] == 2


def test_olivia_plan_uses_work_storage_and_arm64_runtime(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "frank_eq").mkdir(parents=True)
    (root / "configs" / "rate_compute").mkdir(parents=True)
    config = root / "configs" / "rate_compute" / "real_olivia_rc0.yaml"
    config.write_text("run_name: test\n")
    (root / "olivia").mkdir()
    (root / "olivia" / "run.slurm").write_text("#!/bin/bash\n")

    client = ClusterClient("olivia", root=root)
    plan = client.submit(
        job_name="frank-eq-test",
        config_path=config,
        profile="full",
        stages="audit",
        dry_run=True,
    )

    assert plan["remote_root"].startswith("/cluster/work/projects/nn12027k/")
    assert plan["runtime"]["image"].endswith("scratch_pytorch_gcc_updated.sif")
    assert plan["runtime"]["hf_home"] == "/cluster/projects/nn12027k/hf-cache"
    assert plan["runtime"]["partition"] == "accel"
    assert plan["stages"] == "audit"


def test_olivia_scripts_require_arm64_runtime_provenance() -> None:
    root = Path(__file__).parents[1]
    slurm = (root / "olivia" / "run.slurm").read_text()
    smoke = (root / "olivia" / "rc0_runtime_smoke.slurm").read_text()
    quickstart = (root / "olivia" / "quickstart.sh").read_text()

    expected_image = "/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif"
    assert expected_image in slurm
    assert expected_image in smoke
    assert "a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1" in smoke
    assert 'sha256sum "$CONTAINER_IMAGE"' in slurm
    assert 'sha256sum "$SIF_IMAGE"' in smoke
    assert "rate--compute runs require an inspected image SHA-256" in slurm
    assert 'source "$wandb_env"' in slurm
    assert "command -v python3" in quickstart


def test_submit_applies_partition_and_time_overrides(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "frank_eq").mkdir(parents=True)
    (root / "configs" / "stage0").mkdir(parents=True)
    (root / "configs" / "stage0" / "real.yaml").write_text("run_name: test\n")
    (root / "lumi").mkdir()
    (root / "lumi" / "run.slurm").write_text("#!/bin/bash\n")
    monkeypatch.setenv("FRANK_EQ_LUMI_PARTITION", "dev-g")
    monkeypatch.setenv("FRANK_EQ_LUMI_TIME", "02:30:00")
    monkeypatch.setenv("FRANK_EQ_SLURM_DEPENDENCY", "afterok:12345")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-enter-sbatch")
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool = True) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="54321\n", stderr="", returncode=0)

    monkeypatch.setattr("frank_eq.cluster._run", fake_run)
    client = ClusterClient("lumi", root=root)
    submission = client.submit(
        job_name="frank-eq-test",
        config_path="configs/stage0/real.yaml",
        profile="smoke",
        stages="cache,validate",
        dry_run=False,
    )
    assert submission["slurm_job_id"] == "54321"
    assert submission["slurm_partition"] == "dev-g"
    assert submission["slurm_time"] == "02:30:00"
    assert submission["slurm_dependency"] == "afterok:12345"
    submit_command = calls[-1][2]
    assert "--partition=dev-g" in submit_command
    assert "--time=02:30:00" in submit_command
    assert "--dependency=afterok:12345" in submit_command
    assert "FRANK_EQ_STAGES=cache+validate" in submit_command
    assert "FRANK_EQ_STAGES=cache,validate" not in submit_command
    assert "must-not-enter-sbatch" not in submit_command
    setup_command = next(
        command[2]
        for command in calls
        if command[0] == "ssh" and "refusing to overwrite immutable job target" in command[2]
    )
    assert "rm -rf" not in setup_command


def test_submission_dependency_rejects_non_afterok_syntax(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "frank_eq").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs" / "real.yaml").write_text("run_name: test\n")
    (root / "olivia").mkdir()
    (root / "olivia" / "run.slurm").write_text("#!/bin/bash\n")
    monkeypatch.setenv("FRANK_EQ_SLURM_DEPENDENCY", "afterany:12345;touch-bad")

    client = ClusterClient("olivia", root=root)
    with pytest.raises(ValueError, match="single afterok"):
        client.submit(
            job_name="frank-eq-test",
            config_path="configs/real.yaml",
            profile="smoke",
            stages="cache,validate",
            dry_run=True,
        )


def test_source_archive_ignores_local_agent_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "x.py").write_text("x = 1\n")
    archives = root / ".agents" / "state" / "olivia" / ".archives"
    first = build_source_archive(root, archives)
    second = build_source_archive(root, archives)
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == 1


def test_source_archive_ignores_fetched_cluster_results(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "x.py").write_text("x = 1\n")
    fetched = root / ".cluster-results" / "olivia" / "receipt.json"
    fetched.parent.mkdir(parents=True)
    fetched.write_text('{"generated": true}\n')
    archive = build_source_archive(root, tmp_path / "archives")
    assert archive["file_count"] == 1


def test_olivia_recovery_plan_hashes_only_a_failed_predecision_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    config = root / "configs" / "rate_compute" / "real_olivia_rc0.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("run_name: test\n")
    (root / "src" / "frank_eq").mkdir(parents=True)
    (root / "olivia").mkdir()
    (root / "olivia" / "run.slurm").write_text("#!/bin/bash\n")

    source_job = "frank-eq-failed-audit"
    source_state = root / ".agents" / "state" / "olivia" / source_job
    source_runs = source_state / "remote" / "runs"
    (source_runs / "panels").mkdir(parents=True)
    source_submission = {
        "cluster": "olivia",
        "stages": "audit",
        "slurm_job_id": "12345",
        "remote_run_root": (
            "/cluster/work/projects/nn12027k/dakai5365/frank-eq/"
            f"jobs/{source_job}/source/runs"
        ),
        "source": {"sha256": "original-source-sha"},
    }
    (source_state / "submission.json").write_text(json.dumps(source_submission))
    (source_runs / "config.yaml").write_text(config.read_text())
    (source_runs / "workflow_status.json").write_text(
        json.dumps({"state": "failed", "failure": {"type": "ValueError"}})
    )
    for relative in (
        "run_manifest.json",
        "development_splits.json",
        "panels/n4.json",
        "panels/n6.json",
        "models.json",
        "records_raw.jsonl",
        "calibration.json",
        "records_calibrated.jsonl",
    ):
        (source_runs / relative).write_text("{}\n")

    client = ClusterClient("olivia", root=root)
    plan = client.submit(
        job_name="frank-eq-recovered-audit",
        config_path=config,
        profile="full",
        stages="audit",
        dry_run=True,
        recover_from_job=source_job,
    )

    recovery = plan["recovery"]
    assert recovery["source_job_name"] == source_job
    assert recovery["source_slurm_job_id"] == "12345"
    assert recovery["model_capture_executed"] is False
    manifest = json.loads(Path(recovery["input_manifest"]).read_text())
    assert manifest["schema"] == "frank_eq_rate_compute_recovery_input_v1"
    assert set(manifest["files"]) == {
        "config.yaml",
        "run_manifest.json",
        "workflow_status.json",
        "development_splits.json",
        "panels/n4.json",
        "panels/n6.json",
        "models.json",
        "records_raw.jsonl",
        "calibration.json",
        "records_calibrated.jsonl",
    }
