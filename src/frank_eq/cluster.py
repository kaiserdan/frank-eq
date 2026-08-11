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
_DEPENDENCY_PATTERN = re.compile(r"^afterok:[1-9][0-9]*$")
_RATE_COMPUTE_RECOVERY_INPUTS = (
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
)
_RATE_COMPUTE_RECOVERY_FORBIDDEN = (
    "compiled_predictions.jsonl",
    "direct_protocol_selection.json",
    "metrics.json",
    "decision.json",
    "artifact_manifest.json",
    "run_summary.json",
)
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
    partition_environment: str
    time_environment: str


PROFILES = {
    "olivia": ClusterProfile(
        name="olivia",
        host_environment="FRANK_EQ_OLIVIA_HOST",
        default_host="olivia",
        remote_root_environment="FRANK_EQ_OLIVIA_ROOT",
        default_remote_root="/cluster/work/projects/nn12027k/dakai5365/frank-eq",
        slurm_script="olivia/run.slurm",
        partition_environment="FRANK_EQ_OLIVIA_PARTITION",
        time_environment="FRANK_EQ_OLIVIA_TIME",
    ),
    "lumi": ClusterProfile(
        name="lumi",
        host_environment="FRANK_EQ_LUMI_HOST",
        default_host="lumi",
        remote_root_environment="FRANK_EQ_LUMI_ROOT",
        default_remote_root="/scratch/project_465002861/kaiserda/frank-eq",
        slurm_script="lumi/run.slurm",
        partition_environment="FRANK_EQ_LUMI_PARTITION",
        time_environment="FRANK_EQ_LUMI_TIME",
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


def _git_identity(root: Path) -> dict[str, Any]:
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"], check=False)
    status = _run(["git", "-C", str(root), "status", "--porcelain"], check=False)
    return {
        "commit": commit.stdout.strip() if getattr(commit, "returncode", 0) == 0 else None,
        "dirty": (
            bool(status.stdout.strip()) if getattr(status, "returncode", 0) == 0 else None
        ),
    }


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

    def _plan_rate_compute_recovery(
        self,
        *,
        job_name: str,
        source_job_name: str,
        config: Path,
        config_relative: str,
        stages: str,
    ) -> dict[str, Any]:
        if self.profile.name != "olivia":
            raise ValueError("artifact-only RC0 recovery is currently supported only on Olivia")
        self._validate_job_name(source_job_name)
        if source_job_name == job_name:
            raise ValueError("recovery requires a fresh job name")
        if not config_relative.startswith("configs/rate_compute/") or stages != "audit":
            raise ValueError("rate--compute recovery permits exactly --stages audit")
        source_submission = self._load_submission(source_job_name)
        if (
            source_submission.get("cluster") != self.profile.name
            or source_submission.get("stages") != "audit"
        ):
            raise ValueError("recovery source is not an audit from the same cluster")
        source_remote_run_root = str(source_submission.get("remote_run_root", ""))
        expected_prefix = f"{self.remote_root}/jobs/"
        if not source_remote_run_root.startswith(expected_prefix):
            raise ValueError("recovery source is outside the immutable cluster job root")

        source_cache = self._state_dir(source_job_name) / "remote" / "runs"
        missing = [
            relative
            for relative in _RATE_COMPUTE_RECOVERY_INPUTS
            if not (source_cache / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "fetch the complete recovery source first; missing " + ", ".join(missing)
            )
        if any((source_cache / relative).exists() for relative in _RATE_COMPUTE_RECOVERY_FORBIDDEN):
            raise ValueError("recovery source already contains post-calibration outcomes")
        source_status = json.loads((source_cache / "workflow_status.json").read_text())
        if source_status.get("state") != "failed" or not source_status.get("failure"):
            raise ValueError("recovery source is not a failed audit")
        source_config_sha256 = sha256_file(source_cache / "config.yaml")
        if source_config_sha256 != sha256_file(config):
            raise ValueError("recovery config differs from the failed audit")

        recovery_input = {
            "schema": "frank_eq_rate_compute_recovery_input_v1",
            "source_cluster": self.profile.name,
            "source_job_name": source_job_name,
            "source_slurm_job_id": source_submission.get("slurm_job_id"),
            "source_remote_run_root": source_remote_run_root,
            "source_archive_sha256": source_submission.get("source", {}).get("sha256"),
            "source_config_sha256": source_config_sha256,
            "source_failure": source_status["failure"],
            "files": {
                relative: sha256_file(source_cache / relative)
                for relative in _RATE_COMPUTE_RECOVERY_INPUTS
            },
        }
        recovery_input_path = self._state_dir(job_name) / "recovery_input.json"
        atomic_write_json(recovery_input_path, recovery_input)
        return {
            "source_job_name": source_job_name,
            "source_slurm_job_id": source_submission.get("slurm_job_id"),
            "source_remote_run_root": source_remote_run_root,
            "source_archive_sha256": source_submission.get("source", {}).get("sha256"),
            "input_manifest": str(recovery_input_path),
            "input_manifest_sha256": sha256_file(recovery_input_path),
            "remote_input_manifest": (
                f"{self.remote_root}/jobs/{job_name}/source/recovery_input.json"
            ),
            "model_capture_executed": False,
        }

    def plan_submission(
        self,
        *,
        job_name: str,
        config_path: str | Path,
        profile: str,
        stages: str,
        recover_from_job: str | None = None,
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
        recovery = None
        if recover_from_job is not None:
            recovery = self._plan_rate_compute_recovery(
                job_name=job_name,
                source_job_name=recover_from_job,
                config=config,
                config_relative=config_relative,
                stages=stages,
            )
        archive = build_source_archive(self.root, self.state_root / ".archives")
        remote_job = f"{self.remote_root}/jobs/{job_name}"
        git = _git_identity(self.root)
        dependency = os.environ.get("FRANK_EQ_SLURM_DEPENDENCY")
        if dependency and not _DEPENDENCY_PATTERN.fullmatch(dependency):
            raise ValueError(
                "FRANK_EQ_SLURM_DEPENDENCY must be a single afterok:<numeric-job-id> gate"
            )
        if self.profile.name == "lumi":
            runtime = {
                "account": "project_465002861",
                "partition": os.environ.get(self.profile.partition_environment, "small-g"),
                "time": os.environ.get(self.profile.time_environment, "12:00:00"),
                "nodes": 1,
                "gpus_per_node": 1,
                "cpus_per_task": 32,
                "memory": "128G",
                "image": os.environ.get(
                    "FRANK_EQ_LUMI_IMAGE",
                    "/scratch/project_465002861/kaiserda/frank/build_env/usae-deps.sif",
                ),
                "image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
                "hf_home": os.environ.get(
                    "FRANK_EQ_HF_HOME",
                    "/scratch/project_465002861/kaiserda/frank/hf-cache",
                ),
            }
        else:
            runtime = {
                "account": "nn12027k",
                "partition": os.environ.get(self.profile.partition_environment, "accel"),
                "time": os.environ.get(self.profile.time_environment, "12:00:00"),
                "nodes": 1,
                "gpus_per_node": 1,
                "cpus_per_task": 32,
                "memory": "128G",
                "image": os.environ.get(
                    "FRANK_EQ_OLIVIA_IMAGE",
                    "/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif",
                ),
                "image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
                "hf_home": os.environ.get(
                    "FRANK_EQ_HF_HOME",
                    "/cluster/projects/nn12027k/hf-cache",
                ),
            }
        return {
            "schema": "frank_eq_cluster_submission_plan_v1",
            "cluster": self.profile.name,
            "host": self.host,
            "job_name": job_name,
            "profile": profile,
            "stages": stages,
            "config": config_relative,
            "config_sha256": sha256_file(config),
            "source": archive,
            "git": git,
            "runtime": runtime,
            "remote_root": self.remote_root,
            "remote_job": remote_job,
            "remote_source": f"{self.remote_root}/sources/{archive['sha256']}",
            "slurm_script": self.profile.slurm_script,
            "slurm_partition": runtime["partition"],
            "slurm_time": runtime["time"],
            "slurm_dependency": dependency,
            "recovery": recovery,
        }

    def submit(
        self,
        *,
        job_name: str,
        config_path: str | Path,
        profile: str,
        stages: str,
        dry_run: bool,
        recover_from_job: str | None = None,
    ) -> dict[str, Any]:
        plan = self.plan_submission(
            job_name=job_name,
            config_path=config_path,
            profile=profile,
            stages=stages,
            recover_from_job=recover_from_job,
        )
        if dry_run:
            return {**plan, "dry_run": True}
        if (self._state_dir(job_name) / "submission.json").exists():
            raise RuntimeError(
                f"submission state already exists for {job_name}; choose a fresh job name"
            )
        archive_path = plan["source"]["archive"]
        remote_job = plan["remote_job"]
        remote_source = plan["remote_source"]
        remote_archive = f"{remote_job}/source.tar.gz"
        setup = (
            f"if test -e {shlex.quote(remote_job)}; then "
            f"echo {shlex.quote('refusing to overwrite immutable job target')} >&2; exit 73; fi; "
            f"mkdir -p {shlex.quote(self.remote_root)}/sources {shlex.quote(remote_job)}/source"
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
        if plan["recovery"] is not None:
            _run(
                [
                    "rsync",
                    "-a",
                    plan["recovery"]["input_manifest"],
                    f"{self.host}:{plan['recovery']['remote_input_manifest']}",
                ]
            )
        exports = {
            "FRANK_EQ_CLUSTER": self.profile.name,
            "FRANK_EQ_JOB_NAME": job_name,
            "FRANK_EQ_CONFIG": plan["config"],
            "FRANK_EQ_RUN_ROOT": "runs",
            "FRANK_EQ_PROFILE": profile,
            # sbatch --export splits values on commas, so encode the stage list
            # with '+' and decode it again in the job's quickstart script.
            "FRANK_EQ_STAGES": stages.replace(",", "+"),
            "FRANK_EQ_SOURCE_SHA256": plan["source"]["sha256"],
            "FRANK_EQ_GIT_COMMIT": plan["git"]["commit"] or "unknown",
            "FRANK_EQ_GIT_DIRTY": str(plan["git"]["dirty"]).lower(),
        }
        if plan["recovery"] is not None:
            exports.update(
                {
                    "FRANK_EQ_RECOVERY_SOURCE_RUN": plan["recovery"][
                        "source_remote_run_root"
                    ],
                    "FRANK_EQ_RECOVERY_MANIFEST": "recovery_input.json",
                    "FRANK_EQ_RECOVERY_MANIFEST_SHA256": plan["recovery"][
                        "input_manifest_sha256"
                    ],
                }
            )
        for key in (
            "WANDB_ENTITY",
            "WANDB_MODE",
            "WANDB_DIR",
            "WANDB_BASE_URL",
            "FRANK_EQ_LUMI_IMAGE",
            "FRANK_EQ_OLIVIA_IMAGE",
            "FRANK_EQ_IMAGE_SHA256",
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
        ]
        submit_parts.append(f"--partition={plan['slurm_partition']}")
        submit_parts.append(f"--time={plan['slurm_time']}")
        if plan["slurm_dependency"]:
            submit_parts.append(f"--dependency={plan['slurm_dependency']}")
        submit_parts.append(plan["slurm_script"])
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
            "audit": [
                "config.yaml",
                "development_splits.json",
                "panels/n4.json",
                "panels/n6.json",
                "models.json",
                "records_raw.jsonl",
                "calibration.json",
                "records_calibrated.jsonl",
                "compiled_predictions.jsonl",
                "direct_protocol_selection.json",
                "metrics.json",
                "decision.json",
                "artifact_manifest.json",
                "run_summary.json",
            ],
        }
        if submission.get("recovery") is not None:
            expected["audit"].append("recovery_provenance.json")
        artifacts: dict[str, bool] = {}
        for stage in stages:
            for relative in expected.get(stage, []):
                present = (cache_dir / relative).is_file()
                artifacts[relative] = present
                if not present:
                    failures.append(f"missing runs/{relative}")
        scientific_decision = None
        decision_path = (
            cache_dir / "decision.json"
            if stages == ("audit",)
            else cache_dir / "eval" / "decision.json"
        )
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
    submit.add_argument("--recover-from-job")
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
                recover_from_job=args.recover_from_job,
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
