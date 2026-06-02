from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
from configparser import ConfigParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CITATION_CONFIG = ROOT / ".github" / "citation.json"
DEFAULT_CITATION_FILE = ROOT / "CITATION.cff"
DEFAULT_BUMPVERSION_FILE = ROOT / ".bumpversion.cfg"
DEFAULT_SETUP_FILE = ROOT / "setup.py"


def read_version(path: Path) -> str:
    config = ConfigParser()
    config.read(path)
    return config["bumpversion"]["current_version"]


def read_setup_author(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "author" and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
    raise ValueError("setup.py does not define a constant setup(author=...) value")


def read_citation_config(path: Path) -> dict[str, Any]:
    citation_config = json.loads(path.read_text(encoding="utf-8"))
    authors = citation_config.get("authors")
    if not isinstance(authors, list) or not authors:
        raise ValueError("citation config must contain a non-empty authors list")

    for author in authors:
        if not isinstance(author, dict):
            raise ValueError("each author entry must be an object")
        if not author.get("given-names") or not author.get("family-names"):
            raise ValueError("each author must define given-names and family-names")

    return citation_config


def validate_authors(citation_authors: list[dict[str, str]], setup_author: str) -> None:
    configured_names = [
        f"{author['given-names']} {author['family-names']}" for author in citation_authors
    ]
    setup_names = [name.strip() for name in setup_author.split(",") if name.strip()]

    if configured_names != setup_names:
        raise ValueError(
            "citation authors do not match setup.py author list:\n"
            f"citation.json: {configured_names}\n"
            f"setup.py: {setup_names}"
        )


def quote_cff(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_citation(
    citation_config: dict[str, Any],
    version: str,
    release_date: dt.date,
) -> str:
    title = str(citation_config["title"])
    message = str(citation_config["message"])
    authors = citation_config["authors"]

    lines = [
        "cff-version: 1.2.0",
        f"message: {quote_cff(message)}",
        "authors:",
    ]
    for author in authors:
        lines.extend(
            [
                f"  - family-names: {quote_cff(str(author['family-names']))}",
                f"    given-names: {quote_cff(str(author['given-names']))}",
            ]
        )

    lines.extend(
        [
            "",
            f"title: {quote_cff(f'{title}: Version {version}')}",
            f"version: {quote_cff(version)}",
            f"date-released: {quote_cff(release_date.isoformat())}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CITATION.cff for FEZrs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CITATION_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_CITATION_FILE)
    parser.add_argument("--bumpversion", type=Path, default=DEFAULT_BUMPVERSION_FILE)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_FILE)
    parser.add_argument("--date", type=dt.date.fromisoformat)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    citation_config = read_citation_config(args.config)
    validate_authors(citation_config["authors"], read_setup_author(args.setup))

    release_date = args.date or dt.datetime.now(dt.UTC).date()
    content = render_citation(
        citation_config=citation_config,
        version=read_version(args.bumpversion),
        release_date=release_date,
    )

    if args.dry_run:
        print(content, end="")
        return 0

    if args.check:
        current_content = args.output.read_text(encoding="utf-8")
        if current_content != content:
            print(f"{args.output} is out of date")
            return 1
        return 0

    args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
