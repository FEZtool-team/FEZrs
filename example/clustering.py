from pathlib import Path
from fezrs import KMeansCalculator

ROOT = Path(__file__).resolve().parent

KMeansCalculator(
    nir_path=ROOT / "data" / "nir.tif",
    n_clusters=4,
    random_state=0,
).execute(
    output_path=ROOT / "outputs" / "clustering",
)
