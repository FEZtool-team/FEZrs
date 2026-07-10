from fezrs import PCACalculator

PCACalculator(
    blue_path="./data/blue.tif",
    green_path="./data/green.tif",
    red_path="./data/red.tif",
    nir_path="./data/nir.tif",
    swir1_path="./data/swir_1.tif",
    swir2_path="./data/swir_2.tif",
    selectBand="blue",
).execute(
    output_path="./outputs/pca",
).histogram_export(output_path="./outputs/pca")
