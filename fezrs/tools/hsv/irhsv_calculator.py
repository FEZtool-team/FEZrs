# Import packages and libraries
import numpy as np
from skimage.color import rgb2hsv
from typing import Literal

# Import module and files
from fezrs.base import BaseTool, BandPathType

IRHSVChannel = Literal[
    "irhsv",
    "irhue",
    "irsaturation",
    "irvalue",
]


# Calculator class
class IRHSVCalculator(BaseTool):
    def __init__(
        self,
        red_path: BandPathType,
        swir1_path: BandPathType,
        swir2_path: BandPathType,
        channel: IRHSVChannel = "irhsv",
    ):
        super().__init__(
            red_path=red_path,
            swir1_path=swir1_path,
            swir2_path=swir2_path,
        )

        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["red", "swir1", "swir2"]
        )

        self.selected_channel: IRHSVChannel = channel

    def _validate(self):
        pass

    def process(self) -> np.ndarray:
        red, swir1, swir2 = (
            self.normalized_bands[band] for band in ("red", "swir1", "swir2")
        )

        hsv_calculated = rgb2hsv(np.dstack((swir2, swir1, red)))

        channels = {
            "irhsv": hsv_calculated,
            "irhue": hsv_calculated[:, :, 0],
            "irsaturation": hsv_calculated[:, :, 1],
            "irvalue": hsv_calculated[:, :, 2],
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
