from __future__ import annotations

import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize(
    ("distribution", "module"),
    (
        ("yolop-http", "yolop_http"),
        ("yolop-openapi", "yolop_openapi"),
        ("yolop-postgres-runtime", "yolop_postgres_runtime"),
        ("yolop-delegation", "yolop_delegation"),
    ),
)
def test_new_wheel_contains_an_isolated_public_package(
    distribution: str,
    module: str,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[1]
    subprocess.run(
        [
            "uv",
            "build",
            "--package",
            distribution,
            "--wheel",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
    )
    wheel_path = next(tmp_path.glob(f"{distribution.replace('-', '_')}-*.whl"))

    with ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert any(name.startswith(f"{module}/") for name in names)
        metadata_name = next(name for name in names if name.endswith("/METADATA"))
        metadata = wheel.read(metadata_name).decode()
        assert f"Name: {distribution}" in metadata
