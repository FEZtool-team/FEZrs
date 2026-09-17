"""
K-means clustering example.

Paths resolve relative to this file, so the script runs from any directory.
"""

from pathlib import Path

from fezrs import KMeansCalculator

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"

KMeansCalculator(
    nir_path=DATA / "nir.tif", n_clusters=4, random_state=0
).execute(
    output_path=OUTPUTS / "clustering",
)
