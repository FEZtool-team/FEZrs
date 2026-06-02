from pathlib import Path

from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType, SubDivCDType


class SubDivCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        before_nir_path: BandPathType,
        operation: SubDivCDType,
    ):
        super().__init__(
            nir_path=nir_path,
            before_nir_path=before_nir_path,
        )

        self.time_bands = self.files_handler.get_metadata_bands(
            requested_bands=[
                "nir",
                "before_nir",
            ]
        )

        self.operation: SubDivCDType = operation

    def _validate(self):
        pass

    def process(self):

        match self.operation:
            case "divide":
                self._output = (
                    self.time_bands["before_nir"]["image_skimage"]
                    / self.time_bands["nir"]["image_skimage"]
                )
            case "subtract":
                self._output = (
                    self.time_bands["before_nir"]["image_skimage"]
                    - self.time_bands["nir"]["image_skimage"]
                )
            case _:
                self._output = (
                    self.time_bands["before_nir"]["image_skimage"]
                    - self.time_bands["nir"]["image_skimage"]
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
