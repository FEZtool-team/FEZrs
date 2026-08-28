import runpy
import sys
from configparser import ConfigParser
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_setup_uses_runtime_requirements(monkeypatch):
    captured_setup_kwargs = {}
    expected_requirements = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
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
    assert any(
        req.startswith("rasterio>=")
        for req in captured_setup_kwargs["install_requires"]
    )
    assert all(
        "==" not in req for req in captured_setup_kwargs["install_requires"]
    )


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


def test_bumpversion_creates_v_prefixed_tags():
    config = ConfigParser()
    config.read(PROJECT_ROOT / ".bumpversion.cfg")

    assert config.getboolean("bumpversion", "tag") is True
    assert config["bumpversion"]["tag_name"] == "v{new_version}"


def test_pypi_publish_workflow_releases_from_version_tags():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "FEZrs_PyPI_Publish.yml"
    ).read_text(encoding="utf-8")

    assert 'tags:\n      - "v*"' in workflow
    assert "gh release create" in workflow
    assert "startsWith(github.ref, 'refs/tags/v')" in workflow
    assert "bumpversion ${{ inputs.version_bump }}" in workflow
    assert "github.event_name == 'workflow_dispatch' && needs.bump_version.result == 'success'" in workflow
    assert "token: ${{ secrets.GH_PAT }}" not in workflow


def test_test_workflow_runs_on_version_tags():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "FEZrs_Tests.yml"
    ).read_text(encoding="utf-8")

    assert 'tags:\n      - "v*"' in workflow


def test_downstream_publish_workflows_chain_off_pypi_dispatch():
    conda = (
        PROJECT_ROOT / ".github" / "workflows" / "FEZrs_Conda_Publish.yml"
    ).read_text(encoding="utf-8")
    citation = (
        PROJECT_ROOT / ".github" / "workflows" / "FEZrs_Citation_Update.yml"
    ).read_text(encoding="utf-8")

    for workflow in (conda, citation):
        assert "github.event.workflow_run.event == 'push'" in workflow
        assert "github.event.workflow_run.event == 'workflow_dispatch'" in workflow
