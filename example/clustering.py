from fezrs import KMeansCalculator

KMeansCalculator(nir_path="./data/nir.tif", n_clusters=4, random_state=0).execute(
    output_path="./outputs/clustering",
)
