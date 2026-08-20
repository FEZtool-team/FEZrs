"""
Spatial filtering examples.

Paths resolve relative to this file, so the script runs from any directory.
"""

from pathlib import Path

from fezrs import (
    GaussianCalculator,
    LaplacianCalculator,
    MeanCalculator,
    MedianCalculator,
    SobelCalculator,
)

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"

PAN = DATA / "pan_img.tif"
FILTER_OUTPUT = OUTPUTS / "filter"

GaussianCalculator(tif_path=PAN).execute(output_path=FILTER_OUTPUT)

LaplacianCalculator(tif_path=PAN, kernel_size=7).execute(output_path=FILTER_OUTPUT)

MeanCalculator(tif_path=PAN).execute(output_path=FILTER_OUTPUT)

MedianCalculator(tif_path=PAN, kernel_size=5).execute(output_path=FILTER_OUTPUT)

SobelCalculator(tif_path=PAN, kernel_size=7).execute(output_path=FILTER_OUTPUT)
