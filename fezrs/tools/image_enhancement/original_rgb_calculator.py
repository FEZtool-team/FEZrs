# Import packages and libraries
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType
from fezrs.utils.histogram_handler import HistogramExportMixin


# Calculator class
class OriginalRGBCalculator(BaseTool, HistogramExportMixin):
    def __init__(
        self,
        red_path: BandPathType,
        green_path: BandPathType,
        blue_path: BandPathType,
    ):
        super().__init__(
            red_path=red_path,
            blue_path=blue_path,
            green_path=green_path,
        )

        self.normalized_bands = self.files_handler.get_normalized_bands(
            ["red", "green", "blue"]
        )

    def _validate(self):
        pass

    def process(self):
        self._output = np.stack(
            [
                self.normalized_bands["red"],
                self.normalized_bands["green"],
                self.normalized_bands["blue"],
            ],
            axis=2,
        )

        return self._output

    def _customize_export_file(self, ax):
        pass

    def histogram_export(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        filename_prefix: str = "Histogram_RGB_IE_Tool_output",
        dpi: int = 500,
        bbox_inches: str = "tight",
        grid: bool = True,
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
        figsize=(10, 5),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=500,
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


# NOTE - These block code for test the tools, delete before publish product
if __name__ == "__main__":
    red_path = Path.cwd() / "data/Red.tif"
    green_path = Path.cwd() / "data/Green.tif"
    blue_path = Path.cwd() / "data/Blue.tif"

    calculator = OriginalRGBCalculator(
        red_path=red_path, green_path=green_path, blue_path=blue_path
    ).execute("./", title="RGB IE")
