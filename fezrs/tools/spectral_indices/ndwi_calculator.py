# Import packages and libraries
from matplotlib.pyplot import cm

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


# Calculator class
class NDWICalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        green_path: BandPathType,
    ):
        super().__init__(nir_path=nir_path, green_path=green_path)
        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["nir", "green"]
        )

    def _validate(self):
        pass

    def process(self):
        nir, green = (self.normalized_bands[band] for band in ("nir", "green"))

        self._output = (green - nir) / (nir + green)
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap=cm.Grays,
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
