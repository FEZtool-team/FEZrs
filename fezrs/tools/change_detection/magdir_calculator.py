import numpy as np
from pathlib import Path

from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType, MagDirCDType


class MagDirCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        swir1_path: BandPathType,
        before_nir_path: BandPathType,
        before_swir1_path: BandPathType,
        selecte: MagDirCDType,
    ):
        super().__init__(
            nir_path=nir_path,
            swir1_path=swir1_path,
            before_nir_path=before_nir_path,
            before_swir1_path=before_swir1_path,
        )

        self.time_bands = self.files_handler.get_metadata_bands(
            requested_bands=[
                "nir",
                "swir1",
                "before_nir",
                "before_swir1",
            ]
        )

        self.select: MagDirCDType = selecte

    def _validate(self):
        pass

    def process(self):
        after_nir = np.asarray(self.time_bands["nir"]["image_skimage"], dtype=np.float64)
        after_swir1 = np.asarray(
            self.time_bands["swir1"]["image_skimage"], dtype=np.float64
        )
        before_nir = np.asarray(
            self.time_bands["before_nir"]["image_skimage"], dtype=np.float64
        )
        before_swir1 = np.asarray(
            self.time_bands["before_swir1"]["image_skimage"], dtype=np.float64
        )

        delta_nir = after_nir - before_nir
        delta_swir1 = after_swir1 - before_swir1

        change_magnitude_result = np.sqrt(delta_nir**2 + delta_swir1**2)

        # 0 = no change on at least one axis (a difference of exactly 0).
        change_direction_result = np.zeros(after_nir.shape, dtype=np.int8)
        change_direction_result[(delta_nir < 0) & (delta_swir1 < 0)] = 1
        change_direction_result[(delta_nir > 0) & (delta_swir1 < 0)] = 2
        change_direction_result[(delta_nir < 0) & (delta_swir1 > 0)] = 3
        change_direction_result[(delta_nir > 0) & (delta_swir1 > 0)] = 4

        match self.select:
            case "magnitude":
                self._output = change_magnitude_result

            case "direction":
                self._output = change_direction_result

            case _:
                self._output = change_direction_result

        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=True,
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
