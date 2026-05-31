# Import packages and libraries
import numpy as np
from typing import Literal
from skimage.color import rgb2hsv

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType

HSVChannel = Literal[
    "hsv",
    "hue",
    "saturation",
    "value",
]


# Calculator class
class HSVCalculator(BaseTool):
    def __init__(
        self,
        channel: HSVChannel,
        nir_path: BandPathType,
        blue_path: BandPathType,
        green_path: BandPathType,
    ):
        super().__init__(
            nir_path=nir_path,
            blue_path=blue_path,
            green_path=green_path,
        )

        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["nir", "blue", "green"]
        )

        self.selected_channel = channel

    def _validate(self):
        pass

    def process(self) -> np.ndarray:
        nir, blue, green = (
            self.normalized_bands[band] for band in ("nir", "blue", "green")
        )

        hsv_calculated = rgb2hsv(np.dstack((nir, green, blue)))

        channels = {
            "hsv": hsv_calculated,
            "hue": hsv_calculated[:, :, 0],
            "saturation": hsv_calculated[:, :, 1],
            "value": hsv_calculated[:, :, 2],
        }

        self._output = channels[self.selected_channel]

        return self._output

    def _customize_export_file(self, ax):
        pass

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 5),
        show_axis=True,
        colormap=None,
        show_colorbar=True,
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
