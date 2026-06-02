# Import packages and libraries
from matplotlib.pyplot import cm

# Import module and files
from fezrs.base import BaseTool
from fezrs.tools.spectral_indices._division import divide_with_nan
from fezrs.utils.type_handler import BandPathType


# Calculator class
class UICalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        swir2_path: BandPathType,
    ):
        super().__init__(nir_path=nir_path, swir2_path=swir2_path)
        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["nir", "swir2"]
        )

    def _validate(self):
        pass

    def process(self):
        nir, swir2 = (self.normalized_bands[band] for band in ("nir", "swir2"))

        self._output = divide_with_nan(swir2 - nir, nir + swir2)
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap=cm.gray,
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
