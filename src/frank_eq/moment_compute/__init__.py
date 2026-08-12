"""Operation-closed public predictive-state audit for Frank-EQ.

The package installs the prospective Stage-M counterfactual contract before the
workflow module binds its registry and panel builders. This keeps the historical
graph implementation unchanged while making the new experiment explicit and
self-contained.
"""

from . import events as _events
from .config import MomentComputeRunConfig, load_moment_compute_config
from .contract import build_stage_m_event_registry
from .panel import build_moment_panel

_events.build_event_registry = build_stage_m_event_registry

from . import workflow as _workflow  # noqa: E402

_workflow.build_moment_panel = build_moment_panel

from .events import EventRegistry, PublicEvent  # noqa: E402
from .verify import verify_moment_compute_run  # noqa: E402
from .workflow import run_moment_compute_audit, static_contract_summary  # noqa: E402

build_event_registry = build_stage_m_event_registry

__all__ = [
    "EventRegistry",
    "MomentComputeRunConfig",
    "PublicEvent",
    "build_event_registry",
    "load_moment_compute_config",
    "run_moment_compute_audit",
    "static_contract_summary",
    "verify_moment_compute_run",
]
