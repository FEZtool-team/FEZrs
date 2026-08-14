from pathlib import Path
from fezrs import PCACalculator

PCACalculator(
    blue_path= Path.cwd() / "./example/data/blue.tif",
    green_path= Path.cwd() / "./example/data/green.tif",
    red_path= Path.cwd() / "./example/data/red.tif",
    nir_path= Path.cwd() / "./example/data/nir.tif",
    swir1_path= Path.cwd() / "./example/data/swir_1.tif",
    swir2_path= Path.cwd() / "./example/data/swir_2.tif",
    selectBand="blue",
).execute(
    output_path=Path.cwd() /"./example/outputs/pca",
).histogram_export(output_path=Path.cwd() /"./example/outputs/pca")
