# Import module and files
from fezrs.base import BaseTool
from fezrs.tools.spectral_indices._division import divide_with_nan
from fezrs.utils.type_handler import BandPathType


# Calculator class
class SAVICalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        red_path: BandPathType,
    ):
        super().__init__(nir_path=nir_path, red_path=red_path)
        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["nir", "red"]
        )

    def _validate(self):
        pass

    def process(self):
        nir, red = (self.normalized_bands[band] for band in ("nir", "red"))

        self._output = divide_with_nan(nir - red, nir + red + 0.5) * 1.5
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=True,
        filename_prefix="Tool_output",
        dpi=1000,
        bbox_inches="tight",
        grid=True,
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
