# Import packages and libraries
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType
from fezrs.utils.histogram_handler import HistogramExportMixin


# Calculator class
class LogAdjustCalculator(BaseTool, HistogramExportMixin):
    def __init__(
        self,
        nir_path: BandPathType,
        gain: int = 1,
        inverse: bool = False,
    ):
        super().__init__(nir_path=nir_path)

        self.metadata_bands = self.files_handler.get_metadata_bands(["nir"])

        self.gain = gain
        self.inverse = inverse

    def _validate(self):
        pass

    def process(self):
        self._output = exposure.adjust_log(
            img_as_float(self.metadata_bands["nir"]["image_skimage"]),
            gain=self.gain,
            inv=self.inverse,
        )

        return self._output

    def _customize_export_file(self, ax):
        pass

    def histogram_export(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        filename_prefix: str = "Histogram_Log_adjust_IE_Tool_output",
        dpi: int = 500,
        bbox_inches: str = "tight",
    ):
        self._validate()
        self.process()

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            self._output.ravel(),
            bins=256,
            density=True,
            histtype="bar",
            color="black",
        )
        ax.ticklabel_format(style="plain")
        ax.set_title(f"{title}-FEZrs")

        self._add_watermark(ax)
        self._save_histogram_figure(ax, output_path, filename_prefix, dpi, bbox_inches)

        return self

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
