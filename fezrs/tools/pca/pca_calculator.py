from uuid import uuid4
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as skpc
from pathlib import Path

from fezrs.base import BaseTool
from fezrs.utils.histogram_handler import HistogramExportMixin
from fezrs.utils.type_handler import BandNamePCAType, BandPathType


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
        component: int | None = None,
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

        # Raster shape must be (height, width), not (width, height)
        self.image_shape = (
            self.metadata_bands["red"]["height"],
            self.metadata_bands["red"]["width"],
        )

        self.selectBand = selectBand
        self.component = component

        # Legacy name-to-index map used only when selectBand is supplied.
        # These names do not correspond to input bands; they index principal
        # components in collection order. Prefer `component` (1-based).
        self.bindTheBandsToNumber = {
            "red": 0,
            "nir": 1,
            "blue": 2,
            "swir1": 3,
            "swir2": 4,
            "green": 5,
        }

    def _validate(self):
        """
        Validate the input raster bands before performing PCA.
        """

        if len(self.metadata_bands) != 6:
            raise ValueError(
                "PCA requires exactly 6 spectral bands."
            )

        height, width = self.image_shape

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid raster dimensions: "
                f"height={height}, width={width}"
            )

        required_bands = [
            "red",
            "nir",
            "blue",
            "swir1",
            "swir2",
            "green",
        ]

        missing_bands = [
            band
            for band in required_bands
            if band not in self.metadata_bands
        ]

        if missing_bands:
            raise ValueError(
                f"Missing required PCA bands: {missing_bands}"
            )

        component = getattr(self, "component", None)
        if component is not None:
            if not isinstance(component, int) or isinstance(component, bool):
                raise ValueError("component must be an int from 1 to 6.")
            if component < 1 or component > 6:
                raise ValueError("component must be an int from 1 to 6.")

        if self.selectBand is not None:
            if self.selectBand not in self.bindTheBandsToNumber:
                raise ValueError(
                    f"Invalid PCA band: {self.selectBand}"
                )

    def process(self):
        """
        Perform PCA on the six input raster bands.

        Input:
            Six raster images with shape:
                (height, width)

        PCA input matrix:
            (number_of_pixels, number_of_bands)

        Example:
            1000 x 1000 image with 6 bands:

                (1000000, 6)

        PCA output:
            (number_of_pixels, 6)

        Finally, the PCA components are reshaped back into
        raster images:

            (6, height, width)
        """

        self._validate()

        images_collection = self.files_handler.get_images_collection()

        if len(images_collection) != 6:
            raise ValueError(
                "PCA requires exactly 6 input images, "
                f"but received {len(images_collection)}."
            )

        height, width = self.image_shape

        # Validate every raster.
        for index, image in enumerate(images_collection):
            image = np.asarray(image)

            if image.shape != self.image_shape:
                raise ValueError(
                    f"Invalid shape for input band {index}. "
                    f"Expected {self.image_shape}, "
                    f"received {image.shape}."
                )

        images = np.stack(
            [np.asarray(image) for image in images_collection],
            axis=-1,
        )

        images = images.reshape(-1, 6)

        images = images.astype(np.float64, copy=False)

        valid_mask = np.all(
            np.isfinite(images),
            axis=1,
        )

        if not np.any(valid_mask):
            raise ValueError(
                "No valid pixels were found for PCA."
            )

        pca = skpc(n_components=6)

        transformed_valid = pca.fit_transform(
            images[valid_mask]
        )

        transformed = np.full(
            (images.shape[0], 6),
            np.nan,
            dtype=np.float64,
        )

        transformed[valid_mask] = transformed_valid

        self._output = transformed.T.reshape(
            6,
            height,
            width,
        )

        self._pca = pca

        return self._output

    @property
    def explained_variance_ratio(self):
        """Proportion of variance explained by each principal component."""
        pca = getattr(self, "_pca", None)
        if pca is None:
            raise ValueError("PCA has not been computed. Call process() first.")
        return pca.explained_variance_ratio_

    def _resolve_component_index(self) -> int:
        component = getattr(self, "component", None)
        if component is not None:
            return component - 1

        if self.selectBand is None:
            raise ValueError(
                "You cannot use histogram_export() without "
                "passing selectBand."
            )

        warnings.warn(
            "selectBand maps to a principal-component index, not an input "
            f"band. {self.selectBand!r} currently selects component "
            f"{self.bindTheBandsToNumber[self.selectBand] + 1}. "
            "Pass component=N (1-based) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return self.bindTheBandsToNumber[self.selectBand]

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
    ):
        if self.selectBand is None and getattr(self, "component", None) is None:
            raise ValueError(
                "You cannot use histogram_export() without "
                "passing selectBand."
            )

        self._validate()

        if self._output is None:
            self.process()

        component_index = self._resolve_component_index()

        pca_component = self._output[component_index]

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            pca_component.ravel(),
            bins=256,
            density=True,
            histtype="bar",
            color="black",
        )

        ax.set_title(
            f"Histogram of Principal Component {component_index + 1}"
        )

        if title:
            ax.set_title(
                f"{title}-FEZrs"
            )

        ax.set_xlabel("PCA Value")
        ax.set_ylabel("Density")
        ax.grid(grid)

        self._add_watermark(ax)

        self._save_histogram_figure(
            ax,
            output_path,
            filename_prefix,
            dpi,
            bbox_inches,
        )

        plt.close(fig)

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
        """
        Export all six PCA components.

        Each row contains:

            [ PCA image | PCA histogram ]
        """

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
        )

        # Make sure ax is always two-dimensional.
        ax = np.asarray(ax).reshape(
            nrows,
            ncols,
        )

        for i, pca_component in enumerate(self._output):
            # pca_component already has shape:
            #
            #   height x width
            #
            # Therefore, DO NOT reshape it here.
            ax[i, 0].imshow(
                pca_component,
                cmap=colormap,
            )

            ax[i, 0].set_title(
                f"PCA Band {i + 1}"
            )

            if show_axis:
                ax[i, 0].axis("on")
            else:
                ax[i, 0].axis("off")

            ax[i, 1].hist(
                pca_component.ravel(),
                bins=256,
                density=True,
                histtype="bar",
                color="black",
            )

            ax[i, 1].set_title(
                f"Histogram of PCA Band {i + 1}"
            )

            ax[i, 1].grid(grid)

        if title:
            fig.suptitle(title)

        self._customize_export_file(ax)

        filename = (
            f"{output_path}/"
            f"{filename_prefix}_{uuid4().hex}.png"
        )

        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )

        plt.close(fig)

        return self

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