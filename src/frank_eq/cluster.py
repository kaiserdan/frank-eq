"""Content-addressed Olivia/LUMI submission, fetch, and verification helpers."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frank_eq.utils import atomic_write_json, sha256_file

_JOB_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,96}$")
_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "runs",
    "artifacts",
    "checkpoints",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
_EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc", ".pyo"}


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    name: str
    host_environment: str
    default_host: str
    remote_root_environment: str
    default_remote_root: str
    slurm_script: str


PROFILES = {
    "olivia": ClusterProfile(
        name="olivia",
        host_environment="FRANK_EQ_OLIVIA_HOST",
        default_host="olivia",
        remote_root_environment="FRANK_EQ_OLIVIA_ROOT",
        default_remote_root="/cluster/home/dakai5365/project/frank-eq",
        slurm_script="olivia/run.slurm",
    ),
    "lumi": ClusterProfile(
        name="lumi",
        host_environment="FRANK_EQ_LUMI_HOST",
        default_host="lumi",
        remote_root_environment="FRANK_EQ_LUMI_ROOT",
        default_remote_root="/scratch/project_465002861/kaiserda/frank-eq",
        slurm_script="lumi/run.slurm",
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _included_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if len(relative.parts) >= 2 and relative.parts[:2] == (".agents", "state"):
            continue
        if path.suffix in _EXCLUDED_SUFFIXES:
            continue
        if path.name.startswith("slurm-") and path.suffix in {".out", ".err"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_source_archive(root: Path, output_directory: Path) -> dict[str, Any]:
    """Create a deterministic source tarball and content hash."""

    root = root.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    files = _included_files(root)
    with tempfile.NamedTemporaryFile(
        dir=output_directory, prefix="frank-eq-source-", suffix=".tar.gz", delete=False
    ) as temporary:
        archive_path = Path(temporary.name)
    try:
        with (
            archive_path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            for path in files:
                relative = path.relative_to(root).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        digest = sha256_file(archive_path)
        final_path = output_directory / f"{digest}.tar.gz"
        if final_path.exists():
            archive_path.unlink()
        else:
            archive_path.replace(final_path)
        return {
            "archive": str(final_path),
            "sha256": digest,
            "file_count": len(files),
            "bytes": final_path.stat().st_size,
        }
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=True,
    )


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in parts)


class ClusterClient:
    def __init__(self, cluster_name: str, root: Path | None = None):
        try:
            self.profile = PROFILES[cluster_name]
        except KeyError as error:
            raise ValueError(f"unsupported cluster: {cluster_name}") from error
        self.root = (root or repository_root()).resolve()
        self.host = os.environ.get(
            self.profile.host_environment, self.profile.default_host
        )
        self.remote_root = os.environ.get(
            self.profile.remote_root_environment, self.profile.default_remote_root
        ).rstrip("/")
        self.state_root = self.root / ".agents" / "state" / self.profile.name

    def _validate_job_name(self, job_name: str) -> None:
        if not _JOB_PATTERN.fullmatch(job_name):
            raise ValueError(
                "job name must be 3-97 characters containing only letters, digits, '.', '_', '-'"
            )

    def _state_dir(self, job_name: str) -> Path:
        return self.state_root / job_name

    def _load_submission(self, job_name: str) -> dict[str, Any]:
        path = self._state_dir(job_name) / "submission.json"
        if not path.is_file():
            raise FileNotFoundError(f"submission state not found: {path}")
        return json.loads(path.read_text())

    def plan_submission(
        self,
        *,
        job_name: str,
        config_path: str | Path,
        profile: str,
        stages: str,
    ) -> dict[str, Any]:
        self._validate_job_name(job_name)
        if profile not in {"smoke", "full"}:
            raise ValueError("profile must be smoke or full")
        config = Path(config_path)
        if not config.is_absolute():
            config = (self.root / config).resolve()
        if not config.is_file():
            raise FileNotFoundError(f"config not found: {config}")
        try:
            config_relative = config.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("config must be inside the repository source tree") from error
        archive = build_source_archive(self.root, self.state_root / ".archives")
        remote_job = f"{self.remote_root}/jobs/{job_name}"
        return {
            "schema": "frank_eq_cluster_submission_plan_v1",
            "cluster": self.profile.name,
            "host": self.host,
            "job_name": job_name,
            "profile": profile,
            "stages": stages,
            "config": config_relative,
            "source": archive,
            "remote_root": self.remote_root,
            "remote_job": remote_job,
            "remote_source": f"{self.remote_root}/sources/{archive['sha256']}",
            "slurm_script": self.profile.slurm_script,
        }

    def submit(
        self,
        *,
        job_name: str,
        config_path: str | Path,
        profile: str,
        stages: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        plan = self.plan_submission(
            job_name=job_name,
            config_path=config_path,
            profile=profile,
            stages=stages,
        )
        if dry_run:
            return {**plan, "dry_run": True}
        archive_path = plan["source"]["archive"]
        remote_job = plan["remote_job"]
        remote_source = plan["remote_source"]
        remote_archive = f"{remote_job}/source.tar.gz"
        setup = (
            f"mkdir -p {shlex.quote(self.remote_root)}/sources {shlex.quote(remote_job)} && "
            f"rm -rf {shlex.quote(remote_job)}/source && mkdir -p {shlex.quote(remote_job)}/source"
        )
        _run(["ssh", self.host, setup])
        _run(["rsync", "-a", archive_path, f"{self.host}:{remote_archive}"])
        extract = (
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_job)}/source && "
            f"mkdir -p {shlex.quote(remote_source)} && "
            f"cp -f {shlex.quote(remote_archive)} {shlex.quote(remote_source)}/source.tar.gz && "
            f"mkdir -p {shlex.quote(remote_job)}/source/logs"
        )
        _run(["ssh", self.host, extract])
        exports = {
            "FRANK_EQ_CLUSTER": self.profile.name,
            "FRANK_EQ_JOB_NAME": job_name,
            "FRANK_EQ_CONFIG": plan["config"],
            "FRANK_EQ_RUN_ROOT": "runs",
            "FRANK_EQ_PROFILE": profile,
            "FRANK_EQ_STAGES": stages,
            "FRANK_EQ_SOURCE_SHA256": plan["source"]["sha256"],
        }
        for key in (
            "WANDB_API_KEY",
            "WANDB_ENTITY",
            "WANDB_MODE",
            "WANDB_DIR",
            "WANDB_BASE_URL",
            "FRANK_EQ_OLIVIA_IMAGE",
            "FRANK_EQ_HF_HOME",
            "FRANK_EQ_ALLOW_PIP_INSTALL",
            "FRANK_EQ_PIP_FIND_LINKS",
        ):
            value = os.environ.get(key)
            if value:
                exports[key] = value
        export_argument = "ALL," + ",".join(
            f"{key}={value}" for key, value in exports.items()
        )
        submit_parts = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--export={export_argument}",
            plan["slurm_script"],
        ]
        submit_command = f"cd {shlex.quote(remote_job)}/source && " + _shell_join(submit_parts)
        completed = _run(["ssh", self.host, submit_command])
        slurm_job_id = completed.stdout.strip().split(";")[0]
        if not slurm_job_id.isdigit():
            raise RuntimeError(
                f"unable to parse Slurm job ID from {completed.stdout!r}; stderr={completed.stderr!r}"
            )
        submission = {
            **plan,
            "dry_run": False,
            "slurm_job_id": slurm_job_id,
            "remote_run_root": f"{remote_job}/source/runs",
            "remote_logs": f"{remote_job}/source/logs",
        }
        state_dir = self._state_dir(job_name)
        state_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_dir / "submission.json", submission)
        return submission

    def status(self, job_name: str) -> dict[str, Any]:
        submission = self._load_submission(job_name)
        job_id = submission["slurm_job_id"]
        completed = _run(
            [
                "ssh",
                self.host,
                f"sacct -j {shlex.quote(job_id)} --format=JobIDRaw,State,ExitCode,Elapsed -P -n",
            ],
            check=False,
        )
        rows = []
        for line in completed.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 4 and fields[0]:
                rows.append(
                    {
                        "job_id": fields[0],
                        "state": fields[1],
                        "exit_code": fields[2],
                        "elapsed": fields[3],
                    }
                )
        root_row = next((row for row in rows if row["job_id"] == job_id), rows[0] if rows else None)
        remote_status_command = (
            f"test -f {shlex.quote(submission['remote_run_root'])}/workflow_status.json && "
            f"cat {shlex.quote(submission['remote_run_root'])}/workflow_status.json || true"
        )
        remote = _run(["ssh", self.host, remote_status_command], check=False)
        workflow = None
        if remote.stdout.strip():
            try:
                workflow = json.loads(remote.stdout)
            except json.JSONDecodeError:
                workflow = {"state": "unparseable", "raw": remote.stdout[-4000:]}
        payload = {
            "schema": "frank_eq_cluster_status_v1",
            "cluster": self.profile.name,
            "job_name": job_name,
            "slurm_job_id": job_id,
            "scheduler": root_row,
            "scheduler_rows": rows,
            "workflow": workflow,
            "stderr": completed.stderr.strip() or None,
        }
        atomic_write_json(self._state_dir(job_name) / "last_status.json", payload)
        return payload

    def fetch(self, job_name: str) -> dict[str, Any]:
        submission = self._load_submission(job_name)
        state_dir = self._state_dir(job_name)
        cache_dir = state_dir / "remote"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fetched: list[str] = []
        for remote_path, local_name in (
            (submission["remote_run_root"], "runs"),
            (submission["remote_logs"], "logs"),
        ):
            local_path = cache_dir / local_name
            local_path.mkdir(parents=True, exist_ok=True)
            completed = _run(
                ["rsync", "-a", "--partial", f"{self.host}:{remote_path}/", f"{local_path}/"],
                check=False,
            )
            if completed.returncode == 0:
                fetched.append(local_name)
        payload = {
            "schema": "frank_eq_cluster_fetch_v1",
            "cluster": self.profile.name,
            "job_name": job_name,
            "cache_dir": str(cache_dir),
            "fetched": fetched,
        }
        atomic_write_json(state_dir / "fetch.json", payload)
        return payload

    def verify(self, job_name: str) -> dict[str, Any]:
        submission = self._load_submission(job_name)
        cache_dir = self._state_dir(job_name) / "remote" / "runs"
        failures: list[str] = []
        warnings: list[str] = []
        manifest_path = cache_dir / "run_manifest.json"
        status_path = cache_dir / "workflow_status.json"
        if not manifest_path.is_file():
            failures.append("missing runs/run_manifest.json")
        if not status_path.is_file():
            failures.append("missing runs/workflow_status.json")
            workflow = None
        else:
            workflow = json.loads(status_path.read_text())
            if workflow.get("state") != "completed":
                failures.append(f"workflow state is {workflow.get('state')!r}")
        stages = tuple(item for item in submission["stages"].split(",") if item)
        expected = {
            "cache": ["cache/dataset.npz", "cache/metadata.json", "cache/cache_validation.json"],
            "validate": ["cache/cache_validation.json"],
            "train": ["train/final.pt", "train/training_summary.json"],
            "eval": ["eval/metrics.json", "eval/decision.json", "eval/artifact_manifest.json"],
        }
        artifacts: dict[str, bool] = {}
        for stage in stages:
            for relative in expected.get(stage, []):
                present = (cache_dir / relative).is_file()
                artifacts[relative] = present
                if not present:
                    failures.append(f"missing runs/{relative}")
        scientific_decision = None
        decision_path = cache_dir / "eval" / "decision.json"
        if decision_path.is_file():
            scientific_decision = json.loads(decision_path.read_text())
            if scientific_decision.get("status") != "pass":
                warnings.append("scientific gate failed; workflow may still be valid")
        overall = "passed" if not failures else "failed"
        payload = {
            "schema": "frank_eq_cluster_verify_v1",
            "overall": overall,
            "cluster": self.profile.name,
            "job_name": job_name,
            "cache_dir": str(cache_dir.parent),
            "workflow": workflow,
            "artifacts": artifacts,
            "failures": failures,
            "warnings": warnings,
            "scientific_decision": scientific_decision,
            "root_cause": failures[0] if failures else None,
        }
        atomic_write_json(self._state_dir(job_name) / "verify.json", payload)
        return payload


def _print(payload: Any, as_json: bool) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def cluster_cli_main(cluster_name: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"frank-eq-{cluster_name}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--job-name", required=True)
    submit.add_argument("--config", required=True)
    submit.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    submit.add_argument("--stages", default="cache,validate,train,eval")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--json", action="store_true")
    for name in ("status", "fetch", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--job-name", required=True)
        command.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    client = ClusterClient(cluster_name)
    try:
        if args.command == "submit":
            payload = client.submit(
                job_name=args.job_name,
                config_path=args.config,
                profile=args.profile,
                stages=args.stages,
                dry_run=args.dry_run,
            )
        elif args.command == "status":
            payload = client.status(args.job_name)
        elif args.command == "fetch":
            payload = client.fetch(args.job_name)
        elif args.command == "verify":
            payload = client.verify(args.job_name)
        else:
            parser.error("unsupported command")
        _print(payload, args.json)
        return 0 if payload.get("overall", "passed") != "failed" else 2
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"frank-eq-{cluster_name}: {error}", file=os.sys.stderr)
        return 1
