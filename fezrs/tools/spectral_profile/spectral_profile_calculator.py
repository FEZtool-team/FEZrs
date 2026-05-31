# Import packages and libraries
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType
from fezrs.utils.histogram_handler import HistogramExportMixin


class SpectralProfileCalculator(BaseTool, HistogramExportMixin):
    def __init__(
        self,
        red_path: BandPathType,
        green_path: BandPathType,
        blue_path: BandPathType,
        nir_path: BandPathType,
        swir1_path: BandPathType,
        swir2_path: BandPathType,
    ):
        super().__init__(
            red_path=red_path,
            green_path=green_path,
            blue_path=blue_path,
            nir_path=nir_path,
            swir1_path=swir1_path,
            swir2_path=swir2_path,
        )

    def _validate(self):
        pass

    def process(self):
        image_columns = {
            key: value
            for key, value in self.files_handler.bands.items()
            if value is not None
        }
        image_columns_filtered = []
        image_columns_list_of_bands = list(image_columns.keys())

        for key, value in image_columns.items():
            image_columns_filtered.append(value)

        # Ensure imgcol[4] is within bounds and valid
        if len(image_columns_filtered) > 4 and isinstance(
            image_columns_filtered[4], np.ndarray
        ):
            self._output = exposure.adjust_log(image_columns_filtered[4])
        else:
            raise ValueError("Invalid image data at index 4.")

        self.xaxis = []
        self.yaxis = []

        # Prepare x and y axis data for plotting
        for i, img in enumerate(image_columns_filtered):
            self.xaxis.append(f"{image_columns_list_of_bands[i]}")
            self.yaxis.append(np.mean(img))

    def _customize_export_file(self, ax):
        pass

    def histogram_export(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        filename_prefix: str = "Histogram_Spectral_Profile_Tool_output",
        dpi: int = 500,
        bbox_inches: str = "tight",
        grid: bool = True,
        # show_axis: bool = False,
        # colormap: str = None,
        # show_colorbar: bool = False,
    ):
        self._validate()
        self.process()

        fig, ax = plt.subplots(figsize=figsize)

        ax.figure(figsize=figsize)
        ax.plot(self.xaxis, self.yaxis)

        if title:
            plt.title(f"{title}-FEZrs")

        ax.xlabel("Bands")
        ax.ylabel("Intensity")
        ax.grid(grid)

        self._add_watermark(ax)
        self._save_histogram_figure(ax, output_path, filename_prefix, dpi, bbox_inches)

        return self

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=True,
        colormap="gray",
        show_colorbar=False,
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
