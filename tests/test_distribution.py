from importlib import import_module
from importlib.metadata import entry_points, metadata

_CORE_RUNTIME_REQUIREMENTS = {
    "pydantic-ai-slim[spec]>=2.31.0",
    "pyyaml>=6.0.3",
}
_FORBIDDEN_CORE_REQUIREMENT_NAMES = {
    "duckdb",
    "fastapi",
    "mcp",
    "pydantic-ai-harness",
    "pydantic-ai-mcp",
    "pydantic-ai-slim[openai]",
    "sse-starlette",
    "textual",
    "yolop-providers",
    "yolop-tui",
    "yolop-webserver",
}


def test_runtime_distribution_declares_no_host_framework_dependencies() -> None:
    requirements = metadata("yolop-runtime").get_all("Requires-Dist") or []
    base_requirements = {
        requirement for requirement in requirements if "extra ==" not in requirement
    }

    assert base_requirements == {
        "pydantic-ai-slim>=2.31.0",
        "yolop<0.2.0,>=0.1.0",
    }
    forbidden = {"fastapi", "filelock", "sqlite", "textual", "uvicorn"}
    assert not any(
        requirement.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip() in forbidden
        for requirement in base_requirements
    )


def test_core_distribution_declares_only_core_runtime_dependencies() -> None:
    requirements = metadata("yolop").get_all("Requires-Dist") or []
    base_requirements = {
        requirement for requirement in requirements if "extra ==" not in requirement
    }

    assert base_requirements == _CORE_RUNTIME_REQUIREMENTS
    assert not any(
        requirement.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip()
        in _FORBIDDEN_CORE_REQUIREMENT_NAMES
        for requirement in base_requirements
    )


def test_distribution_publishes_current_entry_points() -> None:
    expected = {
        "yolop.capabilities": {
            "DuckDB": "yolop_duckdb:DuckDB",
            "Skills": "yolop.skills:Skills",
            "Workspace": "yolop_workspace:Workspace",
        },
        "yolop.model_providers": {
            "openai-codex": "yolop_providers:create_codex_model",
        },
        "yolop.auth_providers": {
            "openai-codex": "yolop_providers:CodexOAuth",
        },
    }

    for group, group_expected in expected.items():
        actual = {entry_point.name: entry_point.value for entry_point in entry_points(group=group)}
        assert actual == group_expected


def test_public_package_imports_load() -> None:
    public_imports = {
        "yolop": ("ProviderCatalog", "ProviderManifest", "Yolop"),
        "yolop_duckdb": ("DuckDB", "DuckDBDeps"),
        "yolop_providers": ("CodexOAuth", "create_codex_model"),
        "yolop_runtime": ("ExecutionScope", "Runtime", "RuntimeDeps"),
        "yolop_sqlite_session": ("SQLiteRuntimeStore",),
        "yolop_tui": ("run_tui",),
        "yolop_webserver": ("create_app", "RunLimits"),
        "yolop_workspace": ("Workspace", "WorkspaceDeps"),
        "yolop_workspace_session": ("WorkspaceRuntimeStore",),
    }

    for module_name, names in public_imports.items():
        module = import_module(module_name)
        public_names = (*names, *getattr(module, "__all__", ()))
        assert all(hasattr(module, name) for name in public_names)


def test_distribution_publishes_install_profiles() -> None:
    package_metadata = metadata("yolop")

    assert set(package_metadata.get_all("Provides-Extra") or ()) == {
        "all",
        "duckdb",
        "openai",
        "providers",
        "tui",
        "web",
        "workspace",
    }

    requirements = package_metadata.get_all("Requires-Dist") or ()
    expected_profile_dependencies = {
        "duckdb": {"yolop-duckdb"},
        "openai": {"pydantic-ai-slim[openai]"},
        "providers": {"yolop-providers"},
        "tui": {"yolop-tui"},
        "web": {"yolop-webserver"},
        "workspace": {"yolop-workspace"},
        "all": {
            "pydantic-ai-slim[openai]",
            "yolop-duckdb",
            "yolop-providers",
            "yolop-runtime",
            "yolop-sqlite-session",
            "yolop-tui",
            "yolop-webserver",
            "yolop-workspace",
            "yolop-workspace-session",
        },
    }
    for profile, dependencies in expected_profile_dependencies.items():
        actual = {
            requirement.split("<", 1)[0].split(">", 1)[0]
            for requirement in requirements
            if f"extra == '{profile}'" in requirement
        }
        assert actual == dependencies
