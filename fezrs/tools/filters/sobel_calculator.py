# Import packages and libraries
import numpy as np
from cv2 import Sobel

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


class SobelCalculator(BaseTool):
    def __init__(self, tif_path: BandPathType, kernel_size: int):
        super().__init__(tif_path=tif_path)

        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["tif"]
        )

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=["tif"]
        )

        self.kernel_size = kernel_size

    def _validate(self):
        # Validate kernel_size
        if not isinstance(self.kernel_size, int):
            raise TypeError(
                f"'kernel_size' must be an integer, got {type(self.kernel_size).__name__}"
            )
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("'kernel_size' must be a positive odd integer")

        # Validate tif_band
        tif_band = self.files_handler.bands.get("tif")
        if tif_band is None:
            raise ValueError("No 'tif' band found in files_handler.bands")
        if not isinstance(tif_band, np.ndarray):
            raise TypeError(
                f"'tif' band must be a NumPy ndarray, got {type(tif_band).__name__}"
            )
        if tif_band.ndim != 2:
            raise ValueError("'tif' band must be a 2D array")

        # Validate metadata
        metadata = self.metadata_bands.get("tif")
        if not metadata:
            raise ValueError("Missing metadata for 'tif' band")
        if not isinstance(metadata.get("width"), int) or metadata["width"] <= 0:
            raise ValueError("Invalid 'width' in tif metadata")
        if not isinstance(metadata.get("height"), int) or metadata["height"] <= 0:
            raise ValueError("Invalid 'height' in tif metadata")

    def process(self):
        self._output = Sobel(
            self.metadata_bands["tif"]["image_skimage"],
            0,
            dx=1,
            dy=1,
            ksize=self.kernel_size,
        )
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=1000,
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
