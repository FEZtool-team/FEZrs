from pathlib import Path
from fezrs import PCACalculator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "pca"

PCACalculator(
    blue_path=DATA / "blue.tif",
    green_path=DATA / "green.tif",
    red_path=DATA / "red.tif",
    nir_path=DATA / "nir.tif",
    swir1_path=DATA / "swir_1.tif",
    swir2_path=DATA / "swir_2.tif",
    component=1,
).execute(
    output_path=OUT,
).histogram_export(output_path=OUT)
