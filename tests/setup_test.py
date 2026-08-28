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


# --- Declared dependencies match reality (issue #43) --------------------------


def _requirement_names(path):
    """Distribution names from a requirements file, bounds and comments stripped."""
    names = []
    for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for separator in (">=", "==", "<=", "~=", ">", "<"):
            if separator in line:
                line = line.split(separator)[0]
                break
        names.append(line.strip())
    return names


def test_every_runtime_requirement_is_actually_loaded():
    """
    `pydantic` was published as a runtime dependency while appearing nowhere in
    the codebase. Importing the package in a clean interpreter and checking what
    it pulls in catches that, while still allowing genuinely transitive
    dependencies such as imagecodecs, which arrives via scikit-image/tifffile
    and is what decodes compressed GeoTIFFs.
    """
    import subprocess

    # Distribution name -> the top-level module it provides.
    module_of = {
        "Pillow": "PIL",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
        "scikit-image": "skimage",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, fezrs; "
            "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.split())

    unused = [
        name
        for name in _requirement_names("requirements.txt")
        if module_of.get(name, name) not in loaded
    ]

    assert not unused, (
        f"declared as runtime dependencies but never loaded by `import fezrs`: "
        f"{unused}"
    )


def test_lock_file_covers_the_same_distributions_as_requirements():
    """Same distributions, order irrelevant."""
    assert set(_requirement_names("requirements-lock.txt")) == set(
        _requirement_names("requirements.txt")
    )


def test_lock_file_is_exactly_pinned():
    """The lock file is the counterpart to the loosened bounds; it must pin."""
    lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements-lock.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert lines and all("==" in line for line in lines)


def test_dev_requirements_are_test_only():
    """
    gdal, pyrsgis and tensorflow were declared here but imported nowhere, and
    tensorflow alone is several hundred megabytes for anyone following
    CONTRIBUTING.md to run the suite.
    """
    names = _requirement_names("requirements-dev.txt")

    assert "pytest" in names
    for heavy in ("gdal", "pyrsgis", "tensorflow"):
        assert heavy not in names


def test_conda_recipe_covers_every_runtime_dependency():
    """
    The recipe declared fastapi, which the package does not use, and omitted
    pandas, rasterio, pillow and imagecodecs, so `import fezrs` failed after a
    conda install.
    """
    recipe = (PROJECT_ROOT / "recip" / "meta.yaml").read_text(encoding="utf-8")
    run_block = recipe.split("run:", 1)[1].split("about:", 1)[0]
    declared = {
        line.strip().lstrip("- ").split()[0].lower()
        for line in run_block.splitlines()
        if line.strip().startswith("- ")
    }

    # conda channel names differ from the PyPI distribution names.
    conda_name = {
        "opencv-python": "opencv",
        "pillow": "pillow",
    }
    required = {
        conda_name.get(name.lower(), name.lower())
        for name in _requirement_names("requirements.txt")
    }

    assert required <= declared, f"missing from conda recipe: {sorted(required - declared)}"
    assert "fastapi" not in declared


def test_conda_recipe_python_matches_setup_py(monkeypatch):
    recipe = (PROJECT_ROOT / "recip" / "meta.yaml").read_text(encoding="utf-8")

    assert "python >=3.11" in recipe
    assert "python >=3.10" not in recipe


def test_classifiers_cover_the_tested_python_versions(monkeypatch):
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

    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "FEZrs_Tests.yml"
    ).read_text(encoding="utf-8")
    classifiers = captured_setup_kwargs["classifiers"]

    for version in ("3.11", "3.12", "3.13"):
        assert f'"{version}"' in workflow, f"{version} missing from the CI matrix"
        assert (
            f"Programming Language :: Python :: {version}" in classifiers
        ), f"{version} tested but not advertised in setup.py classifiers"
