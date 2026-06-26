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
    author="Mahdi Farmahinifarahani, Hooman Mirzaee, Mahdi Nedaee, Mohammad Hossein Kiani Fayz Abadi, Parsa Elmi",
    author_email="aradfarahani@aol.com",
    description="Feature Extraction and Zoning for Remote Sensing (FEZrs)",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/FEZtool-team/FEZrs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
