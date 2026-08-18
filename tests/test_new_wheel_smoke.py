from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize(
    ("distribution", "module"),
    (
        ("yolop-http", "yolop_http"),
        ("yolop-openapi", "yolop_openapi"),
        ("yolop-postgres-runtime", "yolop_postgres_runtime"),
    ),
)
def test_new_wheel_contains_an_isolated_public_package(
    distribution: str,
    module: str,
) -> None:
    dist = Path(__file__).parents[1] / "dist"
    wheels = sorted(dist.glob(f"{distribution.replace('-', '_')}-*.whl"))
    assert wheels, f"Build {distribution} before running wheel smoke tests"

    with ZipFile(wheels[-1]) as wheel:
        names = set(wheel.namelist())
        assert any(name.startswith(f"{module}/") for name in names)
        metadata_name = next(name for name in names if name.endswith("/METADATA"))
        metadata = wheel.read(metadata_name).decode()
        assert f"Name: {distribution}" in metadata

