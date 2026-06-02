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
