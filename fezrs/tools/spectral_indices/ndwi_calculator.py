# Import packages and libraries
from matplotlib.pyplot import cm

# Import module and files
from fezrs.base import BaseTool
from fezrs.tools.spectral_indices._division import divide_with_nan
from fezrs.utils.radiometry_handler import apply_scaling, warn_if_not_reflectance
from fezrs.utils.type_handler import BandPathType


# Calculator class
class NDWICalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        green_path: BandPathType,
        scale_factor: float = 1.0,
        offset: float = 0.0,
    ):
        super().__init__(nir_path=nir_path, green_path=green_path)
        self.source_bands = apply_scaling(
            self.files_handler.get_bands(requested_bands=["nir", "green"]),
            scale_factor=scale_factor,
            offset=offset,
        )
        warn_if_not_reflectance(self.source_bands, "NDWI")

    def _validate(self):
        pass

    def process(self):
        nir, green = (self.source_bands[band] for band in ("nir", "green"))

        self._output = divide_with_nan(green - nir, nir + green)
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap=cm.Grays,
        show_colorbar=True,
        filename_prefix=None,
        dpi=1000,
        bbox_inches="tight",
        grid=True,
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
