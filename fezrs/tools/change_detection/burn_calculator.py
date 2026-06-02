from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


class BurnCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        swir2_path: BandPathType,
        before_nir_path: BandPathType,
        before_swir2_path: BandPathType,
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

    def _validate(self):
        pass

    def process(self):
        nir = self.time_bands["nir"]["image_skimage"]
        swir2 = self.time_bands["swir2"]["image_skimage"]
        before_nir = self.time_bands["before_nir"]["image_skimage"]
        before_swir2 = self.time_bands["before_swir2"]["image_skimage"]

        indices_after = (nir - swir2) / (nir + swir2)
        indices_before = (before_nir - before_swir2) / (before_nir + before_swir2)
        subtract_before_after = indices_before - indices_after

        self._output = subtract_before_after > 0.7
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
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
