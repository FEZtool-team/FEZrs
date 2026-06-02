import numpy as np
from pathlib import Path
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
            (self.metadata_bands["nir"]["width"], self.metadata_bands["nir"]["height"])
        )

        self.nir_image = np.array(
            self.metadata_bands["nir"]["image_skimage"], dtype="uint8"
        )

        self.property = propery
        self.window_size = window_size

    def process(self):
        for i in range(0, self.metadata_bands["nir"]["width"]):
            print(f"Processing row {i} of {self.metadata_bands['nir']['width']}")
            for j in range(0, self.metadata_bands["nir"]["height"]):
                window = self.nir_image[
                    i : i + self.window_size, j : j + self.window_size
                ]
                glcm = graycomatrix(window, [1], [0], normed=True, symmetric=True)
                res = graycoprops(glcm, self.property)[0][0]
                self.result[i, j] = res
        self._output = self.result

    def _validate(self):
        pass

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
