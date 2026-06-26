import runpy
import sys
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_setup_uses_runtime_requirements(monkeypatch):
    captured_setup_kwargs = {}
    expected_requirements = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setitem(
        sys.modules,
        "setuptools",
        SimpleNamespace(
            setup=lambda **kwargs: captured_setup_kwargs.update(kwargs),
            find_packages=lambda include=None: ["fezrs"],
        ),
    )

    runpy.run_path(str(PROJECT_ROOT / "setup.py"))

    assert captured_setup_kwargs["install_requires"] == expected_requirements
    assert "rasterio==1.5.0" in captured_setup_kwargs["install_requires"]


def test_setup_python_requires_matches_pinned_runtime_dependencies(monkeypatch):
    captured_setup_kwargs = {}

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setitem(
        sys.modules,
        "setuptools",
        SimpleNamespace(
            setup=lambda **kwargs: captured_setup_kwargs.update(kwargs),
            find_packages=lambda include=None: ["fezrs"],
        ),
    )

    runpy.run_path(str(PROJECT_ROOT / "setup.py"))

    assert captured_setup_kwargs["python_requires"] == ">=3.11"
