_EXPORT_MODULES = {
    "AgentLaunchSpec": ".agents",
    "AgentResult": ".agents",
    "AgentSupervisor": ".agents",
    "CodexAppServerRuntime": ".agents",
    "CubeSandboxConfig": ".agents",
    "CubeSandboxRuntime": ".agents",
    "LocalCodexRuntime": ".agents",
    "NetworkPolicy": ".agents",
    "BeamCandidate": ".context",
    "ContextPolicy": ".context",
    "ContextStore": ".context",
    "CrowVerdict": ".context",
    "DiversityPlan": ".context",
    "HelmDirective": ".context",
    "ManagerDecision": ".context",
    "RoleState": ".context",
    "ThoughtSeed": ".context",
    "CubeBootstrapOptions": ".cube",
    "CubeBootstrapper": ".cube",
    "CubeDiagnosis": ".cube",
    "CubeEnvironment": ".cube",
    "BeamSearchOrchestrator": ".orchestrator",
    "DiversityDirector": ".orchestrator",
    "SearchConfig": ".orchestrator",
    "SearchResult": ".orchestrator",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
