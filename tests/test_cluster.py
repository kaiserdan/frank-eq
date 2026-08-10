from pathlib import Path
from types import SimpleNamespace

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


def test_submit_applies_partition_and_time_overrides(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "frank_eq").mkdir(parents=True)
    (root / "configs" / "stage0").mkdir(parents=True)
    (root / "configs" / "stage0" / "real.yaml").write_text("run_name: test\n")
    (root / "lumi").mkdir()
    (root / "lumi" / "run.slurm").write_text("#!/bin/bash\n")
    monkeypatch.setenv("FRANK_EQ_LUMI_PARTITION", "dev-g")
    monkeypatch.setenv("FRANK_EQ_LUMI_TIME", "02:30:00")
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool = True) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="54321\n", stderr="")

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
    submit_command = calls[-1][2]
    assert "--partition=dev-g" in submit_command
    assert "--time=02:30:00" in submit_command
    assert "FRANK_EQ_STAGES=cache+validate" in submit_command
    assert "FRANK_EQ_STAGES=cache,validate" not in submit_command


def test_source_archive_ignores_local_agent_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "x.py").write_text("x = 1\n")
    archives = root / ".agents" / "state" / "olivia" / ".archives"
    first = build_source_archive(root, archives)
    second = build_source_archive(root, archives)
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == 1
