"""
Execute every import statement that appears in the documentation.

Issue #42 found 13 of 23 documented import lines failing, from five distinct
causes -- two of them packaging bugs rather than documentation errors. This test
extracts the import lines from `docs/`, the README and the paper, and runs them,
so documentation and package layout cannot drift apart again.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOC_SOURCES = (
    sorted((PROJECT_ROOT / "docs").glob("*.md"))
    + [PROJECT_ROOT / "README.md"]
    + sorted((PROJECT_ROOT / "paper").glob("*.md"))
)

IMPORT_PATTERN = re.compile(
    r"^\s*(from fezrs[\w.]* import [\w\s,()]+?|import fezrs[\w.]*)\s*$",
    re.MULTILINE,
)


def _read(path: Path) -> str:
    # Windows defaults to cp1252; the docs use UTF-8 box-drawing characters.
    return path.read_text(encoding="utf-8")


def _documented_imports():
    found = []
    for source in DOC_SOURCES:
        if not source.is_file():
            continue
        for match in IMPORT_PATTERN.finditer(_read(source)):
            statement = " ".join(match.group(1).split())
            found.append(pytest.param(statement, id=f"{source.name}::{statement}"))
    return found


DOCUMENTED_IMPORTS = _documented_imports()


def test_documentation_contains_import_examples():
    """Guards against the extraction silently matching nothing."""
    assert len(DOCUMENTED_IMPORTS) >= 15


@pytest.mark.parametrize("statement", DOCUMENTED_IMPORTS)
def test_documented_import_works(statement):
    exec(statement, {})


def test_every_exported_calculator_is_listed_in_the_readme():
    """
    The README listed 28 calculators while the package exported 37, omitting the
    entire change-detection group and SVM.
    """
    import fezrs

    readme = _read(PROJECT_ROOT / "README.md")

    exported = [name for name in fezrs.__all__ if name.endswith("Calculator")]
    # The underscored legacy names stay exported but the README documents the
    # PEP 8 aliases they point at.
    documented = [name for name in exported if "_" not in name]

    missing = [name for name in documented if f"`{name}`" not in readme]

    assert not missing, f"calculators exported but absent from README: {missing}"


def test_subpackages_expose_their_calculators():
    """
    change_detection/ and svm/ had no __init__.py, so nothing could be imported
    from either, and mosaic/__init__.py re-exported BaseTool instead of
    MosaicCalculator.
    """
    from fezrs.tools.change_detection import BurnCalculator  # noqa: F401
    from fezrs.tools.mosaic import MosaicCalculator
    from fezrs.tools.svm import SVMCalculator  # noqa: F401

    assert MosaicCalculator.__name__ == "MosaicCalculator"


def test_media_package_is_importable():
    """
    fezrs/media/ holds the watermark but had no __init__.py, so
    find_packages(include=["fezrs", "fezrs.*"]) in setup.py skipped it.
    """
    import fezrs.media

    assert fezrs.media is not None


def test_setup_find_packages_collects_every_tool_subpackage():
    setuptools = pytest.importorskip("setuptools")
    find_packages = setuptools.find_packages

    packages = set(find_packages(include=["fezrs", "fezrs.*"]))

    for required in (
        "fezrs.media",
        "fezrs.tools.change_detection",
        "fezrs.tools.svm",
        "fezrs.tools.mosaic",
    ):
        assert required in packages, f"{required} would not be packaged"
