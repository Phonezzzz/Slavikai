from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup  # type: ignore[import-untyped]

ROOT = Path(__file__).parent


def _read_requirements() -> list[str]:
    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _read_lock(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


setup(
    name="slavikai",
    version="0.1.0",
    description="SlavikAI local/server agent",
    python_requires=">=3.12",
    packages=find_packages(exclude=("tests", "tests.*", "ui", "ui.*", "vendor", "vendor.*")),
    include_package_data=True,
    install_requires=_read_requirements(),
    extras_require={
        "dev": _read_lock(ROOT / "requirements-dev.txt"),
        "embeddings": _read_lock(ROOT / "requirements-embeddings.txt"),
    },
    entry_points={"console_scripts": ["slavikai=server.http_api:main"]},
)
