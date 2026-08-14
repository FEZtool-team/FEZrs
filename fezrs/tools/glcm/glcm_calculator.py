import numpy as np
from pathlib import Path
from typing import get_args
from fezrs.base import BaseTool
from skimage.feature import graycomatrix, graycoprops
from fezrs.utils.type_handler import BandPathType, PropertyGLCMType


class GLCMCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        window_size: int = 3,
        propery: PropertyGLCMType = "contrast",
    ):
        super().__init__(nir_path=nir_path)

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=["nir"]
        )

        self.result = np.empty(
            (self.metadata_bands["nir"]["height"], self.metadata_bands["nir"]["width"])
        )

        self.nir_image = np.array(
            self.metadata_bands["nir"]["image_skimage"], dtype="uint8"
        )

        self.property = propery
        self.window_size = window_size

    def process(self):
        self._validate()

        height = self.metadata_bands["nir"]["height"]
        width = self.metadata_bands["nir"]["width"]
        for i in range(0, height):
            print(f"Processing row {i} of {height}")
            for j in range(0, width):
                # Window is anchored at the current pixel (i, j) and extends
                # down and to the right. Near the right and bottom edges the
                # slice is clipped to the image, so those pixels use a
                # truncated window instead of padding or being left unset.
                window = self.nir_image[
                    i : i + self.window_size, j : j + self.window_size
                ]
                glcm = graycomatrix(window, [1], [0], normed=True, symmetric=True)
                res = graycoprops(glcm, self.property)[0][0]
                self.result[i, j] = res
        self._output = self.result

    def _validate(self):
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise ValueError("window_size must be an int.")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError(
                "window_size must be an odd integer greater than or equal to 3."
            )

        valid_properties = get_args(PropertyGLCMType)
        if self.property not in valid_properties:
            raise ValueError(
                f"Invalid GLCM property: {self.property!r}. "
                f"Must be one of {list(valid_properties)}."
            )

        files_handler = getattr(self, "files_handler", None)
        if files_handler is None:
            return

        band_paths = getattr(files_handler, "band_paths", None)
        if not isinstance(band_paths, dict):
            return

        nir_path = band_paths.get("nir")
        if not nir_path or not Path(nir_path).is_file():
            raise FileNotFoundError(f"File {nir_path} not found")

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap=None,
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=500,
        bbox_inches="tight",
        grid=False,
        nrows=None,
        ncols=None,
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
            nrows,
            ncols,
        )
