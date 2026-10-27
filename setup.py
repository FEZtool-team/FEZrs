from setuptools import setup, find_packages
from configparser import ConfigParser
from pathlib import Path


ROOT = Path(__file__).parent


def read_version_from_bumpversion():
    config = ConfigParser()
    config.read(ROOT / ".bumpversion.cfg")
    return config["bumpversion"]["current_version"]


def read_requirements(path):
    return [
        line.strip()
        for line in (ROOT / path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="fezrs",
    version=read_version_from_bumpversion(),
    setup_requires=["setuptools", "setuptools_scm"],
    packages=find_packages(include=["fezrs", "fezrs.*"]),
    include_package_data=True,
    package_data={"fezrs": ["media/logo_watermark.png"]},
    install_requires=read_requirements("requirements.txt"),
    author="Mahdi Farmahinifarahani, Parsa Elmi, Mehdi Nedaee, Mohammad Kiani faizabadi, Hooman Mirzaee, Parsa Moradi, Mohammadhossein Yazdanifar, Fariba Khosravani, Fatemeh Najafi, Mehdi Talkhablou",
    author_email="aradfarahani@aol.com",
    # Same one-line summary as the GitHub description and the conda recipe.
    description="Open-source Python package for geospatial multispectral image processing and remote sensing: spectral indices, filters, image enhancement, change detection and classification.",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://feztool.com/fezrs/",
    project_urls={
        "Homepage": "https://feztool.com/fezrs/",
        "Documentation": "https://feztool-team.github.io/FEZdocs/libraries/fezrs/",
        "Source": "https://github.com/FEZtool-team/FEZrs",
        "Issues": "https://github.com/FEZtool-team/FEZrs/issues",
        "Changelog": "https://github.com/FEZtool-team/FEZrs/releases",
        "DOI": "https://doi.org/10.5281/zenodo.14938038",
    },
    license="Apache-2.0",
    keywords=[
        "remote sensing",
        "geospatial",
        "multispectral",
        "satellite imagery",
        "image processing",
        "spectral indices",
        "NDVI",
        "change detection",
        "feature extraction",
        "GLCM",
        "PCA",
        "classification",
        "GeoTIFF",
        "Landsat",
        "earth science",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        # Kept in step with the CI matrix in .github/workflows/FEZrs_Tests.yml,
        # so the advertised support is the support that is actually tested.
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)