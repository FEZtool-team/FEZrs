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
        change_magnitude_result = np.empty(
            self.time_bands["nir"]["image_skimage"].shape
        )
        change_direction_result = np.empty(
            self.time_bands["nir"]["image_skimage"].shape
        )

        for i in range(len(self.time_bands["nir"]["image_skimage"])):
            for j in range(len(self.time_bands["nir"]["image_skimage"][0])):
                change_magnitude = np.sqrt(
                    (
                        self.time_bands["before_nir"]["image_skimage"][i][j]
                        - self.time_bands["nir"]["image_skimage"][i][j]
                    )
                    ** 2
                    + (
                        self.time_bands["before_swir1"]["image_skimage"][i][j]
                        - self.time_bands["swir1"]["image_skimage"][i][j]
                    )
                    ** 2
                )

                if (
                    self.time_bands["nir"]["image_skimage"][i][j]
                    - self.time_bands["before_nir"]["image_skimage"][i][j]
                    < 0
                    and self.time_bands["swir1"]["image_skimage"][i][j]
                    - self.time_bands["before_swir1"]["image_skimage"][i][j]
                    < 0
                ):
                    change_direction = 1
                elif (
                    self.time_bands["nir"]["image_skimage"][i][j]
                    - self.time_bands["before_nir"]["image_skimage"][i][j]
                    > 0
                    and self.time_bands["swir1"]["image_skimage"][i][j]
                    - self.time_bands["before_swir1"]["image_skimage"][i][j]
                    < 0
                ):
                    change_direction = 2
                elif (
                    self.time_bands["nir"]["image_skimage"][i][j]
                    - self.time_bands["before_nir"]["image_skimage"][i][j]
                    < 0
                    and self.time_bands["swir1"]["image_skimage"][i][j]
                    - self.time_bands["before_swir1"]["image_skimage"][i][j]
                    > 0
                ):
                    change_direction = 3
                elif (
                    self.time_bands["nir"]["image_skimage"][i][j]
                    - self.time_bands["before_nir"]["image_skimage"][i][j]
                    > 0
                    and self.time_bands["swir1"]["image_skimage"][i][j]
                    - self.time_bands["before_swir1"]["image_skimage"][i][j]
                    > 0
                ):
                    change_direction = 4

                change_magnitude_result[i][j] = change_magnitude
                change_direction_result[i][j] = change_direction

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
