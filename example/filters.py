from fezrs import (
    GaussianCalculator,
    LaplacianCalculator,
    MeanCalculator,
    MedianCalculator,
    SobelCalculator,
)

GaussianCalculator(
    tif_path="./data/pan_img.tif",
).execute(
    output_path="./outputs/filter",
)

LaplacianCalculator(tif_path="./data/pan_img.tif", kernel_size=7).execute(
    output_path="./outputs/filter",
)

MeanCalculator(tif_path="./data/pan_img.tif").execute(
    output_path="./outputs/filter",
)

MedianCalculator(tif_path="./data/pan_img.tif", kernel_size=5).execute(
    output_path="./outputs/filter",
)

SobelCalculator(tif_path="./data/pan_img.tif", kernel_size=7).execute(
    output_path="./outputs/filter",
)
