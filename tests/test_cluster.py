from pathlib import Path

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


def test_source_archive_ignores_local_agent_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "x.py").write_text("x = 1\n")
    archives = root / ".agents" / "state" / "olivia" / ".archives"
    first = build_source_archive(root, archives)
    second = build_source_archive(root, archives)
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == 1
