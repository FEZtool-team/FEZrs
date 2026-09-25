# Import packages and libraries
import numpy as np
from skimage.color import rgb2hsv
from typing import Literal

# Import module and files
from fezrs.base import BaseTool, BandPathType
from fezrs.utils.nodata_handler import apply_nodata, fill_invalid, invalid_mask

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

        composite = np.dstack((swir2, swir1, red))
        mask = invalid_mask(composite)
        # rgb2hsv turns NaN into ordinary-looking hue and saturation values.
        hsv_calculated = apply_nodata(
            rgb2hsv(fill_invalid(composite, mask) if mask is not None else composite),
            mask,
        )

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
        filename_prefix=None,
        dpi=1000,
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
