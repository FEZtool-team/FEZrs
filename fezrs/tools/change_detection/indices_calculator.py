from pathlib import Path

from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType, TimeCDType


class IndicesCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        swir2_path: BandPathType,
        before_nir_path: BandPathType,
        before_swir2_path: BandPathType,
        time: TimeCDType,
    ):
        super().__init__(
            nir_path=nir_path,
            swir2_path=swir2_path,
            before_nir_path=before_nir_path,
            before_swir2_path=before_swir2_path,
        )

        self.time_bands = self.files_handler.get_metadata_bands(
            requested_bands=[
                "nir",
                "swir2",
                "before_nir",
                "before_swir2",
            ]
        )

        self.selectedTime = time

    def _validate(self):
        pass

    def process(self):
        match self.selectedTime:
            case "after":
                self._output = self._output = (
                    self.time_bands["nir"]["image_skimage"]
                    - self.time_bands["swir2"]["image_skimage"]
                ) / (
                    self.time_bands["nir"]["image_skimage"]
                    + self.time_bands["swir2"]["image_skimage"]
                )
            case "before":
                self._output = (
                    self.time_bands["before_nir"]["image_skimage"]
                    - self.time_bands["before_swir2"]["image_skimage"]
                ) / (
                    self.time_bands["before_nir"]["image_skimage"]
                    + self.time_bands["before_swir2"]["image_skimage"]
                )
            case _:
                self._output = self._output = self._output = (
                    self.time_bands["nir"]["image_skimage"]
                    - self.time_bands["swir2"]["image_skimage"]
                ) / (
                    self.time_bands["nir"]["image_skimage"]
                    + self.time_bands["swir2"]["image_skimage"]
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
