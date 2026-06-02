# Import packages and libraries
import numpy as np
from uuid import uuid4
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skpc

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType, BandNamePCAType
from fezrs.utils.histogram_handler import HistogramExportMixin


class PCACalculator(BaseTool, HistogramExportMixin):
    def __init__(
        self,
        red_path: BandPathType,
        green_path: BandPathType,
        blue_path: BandPathType,
        nir_path: BandPathType,
        swir1_path: BandPathType,
        swir2_path: BandPathType,
        selectBand: BandNamePCAType | None = None,
    ):
        super().__init__(
            red_path=red_path,
            green_path=green_path,
            blue_path=blue_path,
            nir_path=nir_path,
            swir1_path=swir1_path,
            swir2_path=swir2_path,
        )

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=[
                "nir",
                "blue",
                "green",
                "red",
                "swir1",
                "swir2",
            ]
        )

        self.image_shape = (
            self.metadata_bands["red"]["width"],
            self.metadata_bands["red"]["height"],
        )

        self.selectBand = selectBand

        self.bindTheBandsToNumber = {
            "red": 0,
            "nir": 1,
            "blue": 2,
            "swir1": 3,
            "swir2": 4,
            "green": 5,
        }

    def _validate(self):
        pass

    def process(self):
        image_columns = []
        image_columns_filtered = self.files_handler.get_images_collection()
        for i in range(len(image_columns_filtered)):
            image_columns.append(image_columns_filtered[i].flatten())
        images = np.array(image_columns)
        pca = skpc(n_components=6)
        pca.fit(images)
        self._output = pca.components_[:6]
        return self._output

    def _customize_export_file(self, ax):
        pass

    def histogram_export(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        filename_prefix: str = "Histogram_PCA_Tool_output",
        dpi: int = 500,
        bbox_inches: str = "tight",
        grid: bool = True,
        # show_axis: bool = False,
        # colormap: str = None,
        # show_colorbar: bool = False,
    ):
        if self.selectBand is None:
            raise "You cant use histogram method if you are not passed select band value"
        self._validate()
        self.process()

        fig, ax = plt.subplots(figsize=figsize)

        pca_component = self._output[self.bindTheBandsToNumber[self.selectBand]]

        ax.hist(
            pca_component.ravel(),
            bins=256,
            density=True,
            histtype="bar",
            color="black",
        )
        ax.set_title(f"Histogram of PCA Band {self.selectBand.capitalize()}")

        if title:
            plt.title(f"{title}-FEZrs")

        plt.xlabel("Bands")
        plt.ylabel("Intensity")
        plt.grid(grid)

        self._add_watermark(ax)
        self._save_histogram_figure(ax, output_path, filename_prefix, dpi, bbox_inches)

        return self

    def _export_file(
        self,
        output_path,
        title=None,
        figsize=(20, 30),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=1000,
        bbox_inches="tight",
        grid=False,
        nrows=6,
        ncols=2,
    ):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plt.title(title)
        for i, pca_component in enumerate(self._output):
            reshaped_component = pca_component.reshape(self.image_shape)
            ax[i, 0].imshow(reshaped_component, cmap=colormap)
            ax[i, 0].set_title(f"PCA Band {i + 1}")
            ax[i, 0].axis("off")
            ax[i, 1].hist(
                pca_component.ravel(),
                bins=256,
                density=True,
                histtype="bar",
                color="black",
            )
            ax[i, 1].set_title(f"Histogram of PCA Band {i + 1}")

        # Export file
        filename = f"{output_path}/{filename_prefix}_{uuid4().hex}.png"
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

        # Close plt and return value
        plt.close(fig)

    def execute(
        self,
        output_path,
        title=None,
        figsize=(20, 30),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=500,
        bbox_inches="tight",
        grid=True,
        nrows=6,
        ncols=2,
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
