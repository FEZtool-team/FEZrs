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
        after_nir = np.asarray(
            self.time_bands["nir"]["image_skimage"], dtype=np.float64
        )
        after_swir = np.asarray(
            self.time_bands["swir1"]["image_skimage"], dtype=np.float64
        )
        before_nir = np.asarray(
            self.time_bands["before_nir"]["image_skimage"], dtype=np.float64
        )
        before_swir = np.asarray(
            self.time_bands["before_swir1"]["image_skimage"], dtype=np.float64
        )

        d_nir = after_nir - before_nir
        d_swir = after_swir - before_swir

        change_magnitude_result = np.sqrt(d_nir**2 + d_swir**2)

        # 0 = no change (a difference of exactly zero in NIR, SWIR1, or both).
        # The four quadrants require a strict sign in *both* bands; mixed
        # (axis-aligned) change is also 0, rather than inheriting a neighbour.
        change_direction_result = np.zeros(after_nir.shape, dtype=np.int32)
        change_direction_result[(d_nir < 0) & (d_swir < 0)] = 1
        change_direction_result[(d_nir > 0) & (d_swir < 0)] = 2
        change_direction_result[(d_nir < 0) & (d_swir > 0)] = 3
        change_direction_result[(d_nir > 0) & (d_swir > 0)] = 4

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
        filename_prefix=None,
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
