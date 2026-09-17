"""
Principal component analysis example.

Paths resolve relative to this file, so the script runs from any directory.
"""

from pathlib import Path

from fezrs import PCACalculator

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"

calculator = PCACalculator(
    blue_path=DATA / "blue.tif",
    green_path=DATA / "green.tif",
    red_path=DATA / "red.tif",
    nir_path=DATA / "nir.tif",
    swir1_path=DATA / "swir_1.tif",
    swir2_path=DATA / "swir_2.tif",
    component=1,
)

calculator.execute(output_path=OUTPUTS / "pca")
calculator.histogram_export(output_path=OUTPUTS / "pca")

# Variance share and band loadings are what decide which component is worth
# inspecting -- the largest component is not always the informative one.
for index, share in enumerate(calculator.explained_variance_ratio_, start=1):
    print(f"PC{index}: {share:.1%} of total variance")
