# Import packages and libraries
import numpy as np
from sklearn.cluster import KMeans

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


# Calculator class
class KMeansCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        n_clusters: any,
        random_state: any,
    ):
        super().__init__(nir_path=nir_path)

        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["nir"]
        )

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=["nir"]
        )

        self.nir_band = self.files_handler.bands["nir"]

        self.n_clusters = n_clusters
        self.random_state = random_state

    def _validate(self):
        # Validate n_clusters
        if not isinstance(self.n_clusters, int):
            raise TypeError(
                f"Expected 'n_clusters' to be int, got {type(self.n_clusters).__name__}"
            )
        if self.n_clusters < 2:
            raise ValueError(f"'n_clusters' must be >= 2, got {self.n_clusters}")

        # Validate random_state
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(
                f"'random_state' must be an int or None, got {type(self.random_state).__name__}"
            )

        # Validate nir_band
        if not isinstance(self.nir_band, np.ndarray):
            raise TypeError(
                f"NIR band must be a NumPy ndarray, got {type(self.nir_band).__name__}"
            )
        if self.nir_band.ndim != 2:
            raise ValueError("NIR band must be a 2D array")

        # Validate metadata for NIR band
        metadata = self.metadata_bands.get("nir")
        if not metadata:
            raise ValueError("Missing metadata for 'nir' band")
        if not isinstance(metadata.get("width"), int) or metadata["width"] <= 0:
            raise ValueError("Invalid width in NIR metadata")
        if not isinstance(metadata.get("height"), int) or metadata["height"] <= 0:
            raise ValueError("Invalid height in NIR metadata")

    def process(self) -> np.ndarray:
        image_reshape = self.nir_band.reshape(
            self.metadata_bands["nir"]["width"] * self.metadata_bands["nir"]["height"],
            1,
        )

        # Define the number of clusters (you can change this number depending on your needs)
        # These variables are extra parameters
        n_clusters = self.n_clusters
        random_state = self.random_state

        # Initialize and fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(image_reshape)

        # Get cluster centers and labels
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        # Reshape the clustered labels back into the image dimensions
        clusterd_image = cluster_centers[cluster_labels].reshape(
            self.metadata_bands["nir"]["height"], self.metadata_bands["nir"]["width"]
        )

        self._output = clusterd_image

        return self._output

    def _customize_export_file(self, ax):
        pass

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 15),
        show_axis=False,
        colormap=None,
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=500,
        bbox_inches="tight",
        grid=False,
    ):
        return super().execute(
            output_path,
            title,
            figsize,
            show_axis,
            colormap,
            show_colorbar,
            filename_prefix,
            dpi,
            bbox_inches,
            grid,
        )
