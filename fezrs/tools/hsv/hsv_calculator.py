# Import packages and libraries
import numpy as np
from typing import Literal
from skimage.color import rgb2hsv

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.nodata_handler import apply_nodata, fill_invalid, invalid_mask
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

        composite = np.dstack((nir, green, blue))
        mask = invalid_mask(composite)
        # rgb2hsv turns NaN into ordinary-looking hue and saturation values.
        hsv_calculated = apply_nodata(
            rgb2hsv(fill_invalid(composite, mask) if mask is not None else composite),
            mask,
        )

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
