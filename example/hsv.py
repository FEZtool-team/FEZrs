from fezrs import HSVCalculator

HSVCalculator(
    blue_path="./data/blue.tif",
    green_path="./data/green.tif",
    nir_path="./data/nir.tif",
    channel="hsv",
).execute(
    output_path="./outputs/hsv",
    show_axis=False,
    show_colorbar=False,
)
