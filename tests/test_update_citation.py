import datetime as dt
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "update_citation.py"


def load_update_citation_module():
    spec = importlib.util.spec_from_file_location("update_citation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_citation_uses_configured_authors_and_version():
    update_citation = load_update_citation_module()
    citation_config = {
        "title": "FEZrs",
        "message": "Please cite FEZrs.",
        "authors": [
            {
                "given-names": "Mahdi",
                "family-names": "Farmahinifarahani",
            }
        ],
    }

    result = update_citation.render_citation(
        citation_config=citation_config,
        version="1.2.3",
        release_date=dt.date(2026, 6, 2),
    )

    assert result == (
        "cff-version: 1.2.0\n"
        'message: "Please cite FEZrs."\n'
        "authors:\n"
        '  - family-names: "Farmahinifarahani"\n'
        '    given-names: "Mahdi"\n'
        "\n"
        'title: "FEZrs: Version 1.2.3"\n'
        'version: "1.2.3"\n'
        'date-released: "2026-06-02"\n'
    )


def test_validate_authors_rejects_setup_mismatch():
    update_citation = load_update_citation_module()
    citation_authors = [
        {
            "given-names": "Mahdi",
            "family-names": "Farmahinifarahani",
        }
    ]

    with pytest.raises(ValueError, match="citation authors do not match"):
        update_citation.validate_authors(citation_authors, "Another Author")


def test_script_uses_python310_compatible_utc_timezone():
    assert "dt.UTC" not in SCRIPT_PATH.read_text(encoding="utf-8")


def test_render_citation_emits_optional_metadata():
    update_citation = load_update_citation_module()
    citation_config = {
        "title": "FEZrs",
        "message": "Please cite FEZrs.",
        "authors": [
            {
                "given-names": "Mahdi",
                "family-names": "Farmahinifarahani",
                "orcid": "https://orcid.org/0009-0008-3800-8688",
            }
        ],
        "doi": "10.5281/zenodo.14938038",
        "license": "Apache-2.0",
        "keywords": ["remote sensing", "NDVI"],
    }

    result = update_citation.render_citation(
        citation_config=citation_config,
        version="1.2.3",
        release_date=dt.date(2026, 6, 2),
    )

    assert result == (
        "cff-version: 1.2.0\n"
        'message: "Please cite FEZrs."\n'
        "authors:\n"
        '  - family-names: "Farmahinifarahani"\n'
        '    given-names: "Mahdi"\n'
        '    orcid: "https://orcid.org/0009-0008-3800-8688"\n'
        "\n"
        'title: "FEZrs: Version 1.2.3"\n'
        'version: "1.2.3"\n'
        'date-released: "2026-06-02"\n'
        'doi: "10.5281/zenodo.14938038"\n'
        'license: "Apache-2.0"\n'
        "keywords:\n"
        '  - "remote sensing"\n'
        '  - "NDVI"\n'
    )


def test_read_citation_config_rejects_bare_orcid(tmp_path):
    update_citation = load_update_citation_module()
    config = tmp_path / "citation.json"
    config.write_text(
        '{"title": "FEZrs", "message": "m", "authors": ['
        '{"given-names": "Mahdi", "family-names": "Farmahinifarahani",'
        ' "orcid": "0009-0008-3800-8688"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="orcid"):
        update_citation.read_citation_config(config)


def test_committed_citation_matches_its_config():
    update_citation = load_update_citation_module()
    root = SCRIPT_PATH.parents[2]
    citation_config = update_citation.read_citation_config(root / ".github" / "citation.json")
    update_citation.validate_authors(
        citation_config["authors"],
        update_citation.read_setup_author(root / "setup.py"),
    )
    committed = (root / "CITATION.cff").read_text(encoding="utf-8")
    released = committed.split('date-released: "', 1)[1].split('"', 1)[0]

    assert committed == update_citation.render_citation(
        citation_config=citation_config,
        version=update_citation.read_version(root / ".bumpversion.cfg"),
        release_date=dt.date.fromisoformat(released),
    )
