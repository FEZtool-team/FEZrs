"""
HSV colour-space example.

Paths resolve relative to this file, so the script runs from any directory.
"""

from pathlib import Path

from fezrs import HSVCalculator

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"

HSVCalculator(
    blue_path=DATA / "blue.tif",
    green_path=DATA / "green.tif",
    nir_path=DATA / "nir.tif",
    channel="hsv",
).execute(
    output_path=OUTPUTS / "hsv",
    show_axis=False,
    show_colorbar=False,
)
