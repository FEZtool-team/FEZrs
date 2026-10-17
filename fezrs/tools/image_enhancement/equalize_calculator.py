# Import packages and libraries
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.nodata_handler import apply_nodata, fill_invalid, invalid_mask
from fezrs.utils.type_handler import BandPathType
from fezrs.utils.histogram_handler import HistogramExportMixin


# Calculator class
class EqualizeCalculator(BaseTool, HistogramExportMixin):
    def __init__(self, nir_path: BandPathType):
        super().__init__(nir_path=nir_path)

        self.metadata_bands = self.files_handler.get_metadata_bands(["nir"])

    def _validate(self):
        pass

    def process(self):
        image = img_as_float(self.metadata_bands["nir"]["image_skimage"])
        mask = invalid_mask(image)

        if mask is None:
            self._output = exposure.equalize_hist(image, nbins=256, mask=None)
        else:
            # Build the histogram from valid pixels only; NaN cannot be binned.
            equalized = exposure.equalize_hist(
                fill_invalid(image, mask), nbins=256, mask=~mask
            )
            self._output = apply_nodata(equalized, mask)

        return self._output

    def _customize_export_file(self, ax):
        pass

    def histogram_export(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        filename_prefix: str = "Histogram_Equlize_IE_Tool_output",
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
