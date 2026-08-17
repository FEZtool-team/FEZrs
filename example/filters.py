from pathlib import Path
from fezrs import (
    GaussianCalculator,
    LaplacianCalculator,
    MeanCalculator,
    MedianCalculator,
    SobelCalculator,
)

ROOT = Path(__file__).resolve().parent
PAN = ROOT / "data" / "pan_img.tif"
OUT = ROOT / "outputs" / "filter"

GaussianCalculator(
    tif_path=PAN,
).execute(
    output_path=OUT,
)

LaplacianCalculator(tif_path=PAN, kernel_size=7).execute(
    output_path=OUT,
)

MeanCalculator(tif_path=PAN).execute(
    output_path=OUT,
)

MedianCalculator(tif_path=PAN, kernel_size=5).execute(
    output_path=OUT,
)

SobelCalculator(tif_path=PAN, kernel_size=7).execute(
    output_path=OUT,
)
