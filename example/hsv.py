from pathlib import Path
from fezrs import HSVCalculator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

HSVCalculator(
    blue_path=DATA / "blue.tif",
    green_path=DATA / "green.tif",
    nir_path=DATA / "nir.tif",
    channel="hsv",
).execute(
    output_path=ROOT / "outputs" / "hsv",
    show_axis=False,
    show_colorbar=False,
)
