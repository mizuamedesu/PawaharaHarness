from .agents import (
    AgentLaunchSpec,
    AgentResult,
    AgentSupervisor,
    CodexAppServerRuntime,
    CubeSandboxConfig,
    CubeSandboxRuntime,
    LocalCodexRuntime,
    NetworkPolicy,
)
from .context import (
    BeamCandidate,
    ContextPolicy,
    ContextStore,
    CrowVerdict,
    DiversityPlan,
    HelmDirective,
    ManagerDecision,
    RoleState,
    ThoughtSeed,
)
from .orchestrator import BeamSearchOrchestrator, DiversityDirector, SearchConfig, SearchResult

_LAZY_CUBE_EXPORTS = {
    "CubeBootstrapOptions",
    "CubeBootstrapper",
    "CubeDiagnosis",
    "CubeEnvironment",
}

__all__ = [
    "AgentLaunchSpec",
    "AgentResult",
    "AgentSupervisor",
    "BeamCandidate",
    "BeamSearchOrchestrator",
    "CodexAppServerRuntime",
    "ContextPolicy",
    "ContextStore",
    "CrowVerdict",
    "CubeSandboxConfig",
    "CubeSandboxRuntime",
    "HelmDirective",
    "LocalCodexRuntime",
    "NetworkPolicy",
    "DiversityDirector",
    "DiversityPlan",
    "ManagerDecision",
    "RoleState",
    "SearchConfig",
    "SearchResult",
    "ThoughtSeed",
    "CubeBootstrapOptions",
    "CubeBootstrapper",
    "CubeDiagnosis",
    "CubeEnvironment",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_CUBE_EXPORTS:
        from . import cube

        value = getattr(cube, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
