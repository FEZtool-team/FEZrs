import warnings
from uuid import uuid4

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
        standardize: bool = False,
    ):
        """
        Args:
            component: Principal component to inspect, numbered 1..6 in order of
                decreasing explained variance.
            selectBand: Deprecated. A band name that resolves to a fixed
                component index. A principal component is a linear combination
                of all six input bands, so no component corresponds to an input
                band; use ``component`` instead.
            standardize: Run PCA on the correlation matrix rather than the
                covariance matrix, by scaling each band to unit variance first.
                Without it, bands with the largest digital-number range dominate
                the leading components regardless of their information content.
        """
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

        self.standardize = standardize

        # Input band order.
        #
        # This order MUST match the order returned by
        # files_handler.get_images_collection().
        self.bindTheBandsToNumber = {
            "red": 0,
            "nir": 1,
            "blue": 2,
            "swir1": 3,
            "swir2": 4,
            "green": 5,
        }

        self.band_order = [
            name
            for name, _ in sorted(
                self.bindTheBandsToNumber.items(), key=lambda item: item[1]
            )
        ]

        if component is not None and selectBand is not None:
            raise ValueError("Pass either 'component' or 'selectBand', not both.")

        if selectBand is not None:
            if selectBand not in self.bindTheBandsToNumber:
                raise ValueError(f"Invalid PCA band: {selectBand}")
            resolved = self.bindTheBandsToNumber[selectBand] + 1
            warnings.warn(
                f"PCACalculator's 'selectBand' is deprecated: {selectBand!r} "
                f"resolves to principal component {resolved}. A principal "
                "component is a linear combination of all six input bands, so "
                "no component corresponds to an input band. Pass "
                f"component={resolved} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            component = resolved

        if component is not None and not 1 <= component <= 6:
            raise ValueError(
                f"component must be between 1 and 6, received {component}."
            )

        self.component = component
        self.selectBand = selectBand
        self._pca = None

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

        if self.component is not None and not 1 <= self.component <= 6:
            raise ValueError(
                f"component must be between 1 and 6, received {self.component}."
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

        valid_images = images[valid_mask]

        if self.standardize:
            # Correlation-matrix PCA. sklearn decomposes the covariance matrix,
            # which lets whichever band happens to carry the widest
            # digital-number range dominate the leading components regardless of
            # how much information it holds.
            band_std = valid_images.std(axis=0)
            band_std[band_std == 0] = 1.0
            valid_images = (valid_images - valid_images.mean(axis=0)) / band_std

        pca = skpc(n_components=6)

        transformed_valid = pca.fit_transform(valid_images)

        # Eigenvector signs are arbitrary: the same scene can yield an inverted
        # component image between runs, or between a scene and a crop of it,
        # which makes results non-reproducible and the imagery hard to read.
        # Fix the convention so the largest-magnitude loading is always
        # positive.
        dominant = np.argmax(np.abs(pca.components_), axis=1)
        signs = np.sign(pca.components_[np.arange(6), dominant])
        signs[signs == 0] = 1.0

        pca.components_ = pca.components_ * signs[:, np.newaxis]
        transformed_valid = transformed_valid * signs

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
    def explained_variance_ratio_(self):
        """
        Share of total variance carried by each principal component.

        Returns:
            np.ndarray: Shape ``(6,)``, ordered by decreasing variance.

        Raises:
            RuntimeError: If ``process()`` has not run yet.
        """
        if self._pca is None:
            raise RuntimeError(
                "Run process() before reading explained_variance_ratio_."
            )
        return self._pca.explained_variance_ratio_

    @property
    def components_(self):
        """
        Eigenvector loadings, shape ``(6, 6)`` as ``(component, band)``.

        Band order is given by ``band_order``. These loadings, not the variance
        share, are what identify a useful component: a target is isolated by the
        component in which the relevant bands carry high loadings of opposing
        sign.

        Signs follow a fixed convention (largest-magnitude loading positive), so
        repeated runs and crops of the same scene are directly comparable.

        Raises:
            RuntimeError: If ``process()`` has not run yet.
        """
        if self._pca is None:
            raise RuntimeError("Run process() before reading components_.")
        return self._pca.components_

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
        

        if self.component is None:
            raise ValueError(
                "You cannot use histogram_export() without passing component "
                "(or the deprecated selectBand)."
            )

        self._validate()

        if not hasattr(self, "_output"):
            self.process()

        component_index = self.component - 1

        pca_component = self._output[component_index]

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            pca_component.ravel(),
            bins=256,
            density=True,
            histtype="bar",
            color="black",
        )

        # Labelling this by an input band name attributed the output to a band
        # that did not produce it: every component mixes all six inputs.
        ax.set_title(
            f"Histogram of Principal Component {self.component}"
        )

        if title:
            ax.set_title(
                f"{title}-FEZrs"
            )

        ax.set_xlabel(f"PC{self.component} Value")
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
        filename_prefix=None,
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
        filename_prefix=None,
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