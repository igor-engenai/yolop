from importlib.metadata import metadata


def test_distribution_publishes_install_profiles() -> None:
    package_metadata = metadata("yolop")

    assert set(package_metadata.get_all("Provides-Extra") or ()) == {
        "all",
        "duckdb",
        "openai",
        "tui",
        "web",
        "workspace",
    }

    requirements = package_metadata.get_all("Requires-Dist") or ()
    expected_profile_dependencies = {
        "duckdb": {"yolop-duckdb"},
        "openai": {"pydantic-ai-slim[openai]"},
        "tui": {"yolop-tui"},
        "web": {"yolop-webserver"},
        "workspace": {"yolop-workspace"},
        "all": {
            "pydantic-ai-slim[openai]",
            "yolop-duckdb",
            "yolop-session",
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
