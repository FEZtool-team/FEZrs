# Import packages and libraries
from abc import ABC

import numpy as np
import rasterio as rio
from PIL import Image
from pathlib import Path
from uuid import uuid4
from importlib import resources
import matplotlib.pyplot as plt


# Import module and files
from fezrs.utils.file_handler import FileHandler
from fezrs.utils.type_handler import BandPathType, BandPathsType


# Definition abstract class (BaseTool)
class BaseTool(ABC):
    """
    Abstract base class for FEZrs tools.

    Provides common initialization, validation, processing, and export logic for derived tools.
    Handles band file paths, watermarking, and standardized export of results.
    """

    def __init__(self, **bands_path: BandPathsType):
        """
        Initializes the BaseTool with band file paths and loads the watermark logo.

        Args:
            **bands_path: Arbitrary keyword arguments representing band file paths.
        """
        self._output = None
        self.__tool_name = self.__class__.__name__.replace("Calculator", "")

        logo_resource = resources.files("fezrs.media").joinpath("logo_watermark.png")
        with resources.as_file(logo_resource) as logo_path:
            logo_img = Image.open(logo_path).convert("RGBA")
            logo_img = logo_img.resize((80, 80))

        self._logo_watermark = logo_img

        self.files_handler = FileHandler(**bands_path)

    def _validate(self):
        """
        Abstract method for validating input data or configuration.

        Should be implemented by subclasses to perform tool-specific validation.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def process(self):
        """
        Abstract method for processing data.

        Should be implemented by subclasses to perform the main computation.
        """
        self._validate()
        raise NotImplementedError("Subclasses should implement this method")

    def _customize_export_file(self, ax):
        """
        Hook for subclasses to customize the export plot.

        Args:
            ax: The matplotlib axes object to customize.
        """
        pass

    def _export_file(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        show_axis: bool = False,
        colormap: str = None,
        show_colorbar: bool = False,
        filename_prefix: str | None = None,
        dpi: int = 500,
        bbox_inches: str = "tight",
        grid: bool = True,
        nrows: int = 1,
        ncols: int = 1,
    ):
        """
        Exports the computed output as a PNG image with optional customization.

        Args:
            output_path: Directory to save the exported image.
            title: Optional title for the plot.
            figsize: Figure size for the plot.
            show_axis: Whether to display axes.
            colormap: Colormap for the image.
            show_colorbar: Whether to display a colorbar.
            filename_prefix: Prefix for the output filename.
            dpi: Dots per inch for the saved image.
            bbox_inches: Bounding box option for saving the figure.
            grid: Whether to display a grid.
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.

        Returns:
            The path to the saved image file.
        """
        # Fall back to the tool name only when the caller did not choose a
        # prefix. Previously the argument was accepted, documented, and then
        # overwritten on this line before it was ever used.
        if filename_prefix is None:
            filename_prefix = self.__tool_name

        # Check output property is not empty
        if self._output is None:
            raise ValueError("Data not computed.")

        # Check the output path is exist and if not create that directory(ies)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run plot methods
        fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        im = ax.imshow(self._output, cmap=colormap)
        plt.grid(grid)

        # Arguments conditions
        if not show_axis:
            ax.axis("off")

        if show_colorbar:
            fig.colorbar(im, ax=ax)

        if title:
            plt.title(f"{title}-FEZrs")

        self._customize_export_file(ax)

        # Export file
        filename = f"{output_path}/{filename_prefix}_output_{uuid4().hex}.png"
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

        # Close plt and return value
        plt.close(fig)
        return filename

    def to_raster(
        self,
        output_path: BandPathType,
        reference_band: str | None = None,
        dtype: str | None = None,
        nodata=None,
        compress: str = "deflate",
    ):
        """
        Write the computed result as a georeferenced GeoTIFF.

        ``execute()`` renders through matplotlib and saves a PNG: values are
        quantized to 256 levels per channel, the pixel grid is resampled by dpi
        and bbox_inches, and CRS and transform are discarded. That output is a
        picture of the result, not the result. This method writes the array
        itself, carrying the CRS and affine transform of the source band, so it
        can be overlaid in a GIS, intersected with mapped units, differenced
        against another date, or used for zonal statistics.

        Multi-component outputs, such as PCA's ``(6, height, width)``, are
        written as multi-band rasters.

        Args:
            output_path: Destination ``.tif`` path. Parent directories are
                created.
            reference_band: Band whose spatial referencing to copy. Defaults to
                the first supplied band.
            dtype: Output dtype. Defaults to ``float32`` for continuous results
                and ``int32`` for integer label maps such as classifications.
                float32 is well beyond reflectance precision and half the size
                of float64.
            nodata: Nodata value. Defaults to NaN for floating point output.
            compress: GeoTIFF compression. ``deflate`` with a horizontal
                differencing predictor is the usual choice for float rasters.

        Returns:
            str: Path to the written raster.

        Raises:
            ValueError: If nothing has been computed, or the source carried no
                spatial referencing to propagate.
        """
        if self._output is None:
            raise ValueError("Data not computed.")

        profile = self.files_handler.get_raster_profile(reference_band)

        if profile is None or profile["crs"] is None:
            raise ValueError(
                "The source band carries no CRS, so the result cannot be "
                "written as a georeferenced raster. Reading the inputs from "
                "GeoTIFFs that declare a CRS and transform will fix this. Use "
                "execute() if a plain image is what you want."
            )

        array = np.asarray(self._output)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        elif array.ndim != 3:
            raise ValueError(
                f"Cannot write an array with {array.ndim} dimensions as a raster."
            )

        if dtype is None:
            dtype = (
                "int32" if np.issubdtype(array.dtype, np.integer) else "float32"
            )

        if nodata is None and not np.issubdtype(np.dtype(dtype), np.integer):
            nodata = float("nan")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        creation = {
            "driver": "GTiff",
            "height": array.shape[1],
            "width": array.shape[2],
            "count": array.shape[0],
            "dtype": dtype,
            "crs": profile["crs"],
            "transform": profile["transform"],
            "nodata": nodata,
            "tiled": True,
            "compress": compress,
            "BIGTIFF": "IF_SAFER",
        }
        if compress == "deflate" and not np.issubdtype(np.dtype(dtype), np.integer):
            # Horizontal differencing for floating point data.
            creation["predictor"] = 3

        with rio.open(output_path, "w", **creation) as destination:
            destination.write(array.astype(dtype))

        return str(output_path)

    def execute(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        show_axis: bool = False,
        colormap: str = None,
        show_colorbar: bool = False,
        filename_prefix: str | None = None,
        dpi: int = 500,
        bbox_inches: str = "tight",
        grid: bool = True,
        nrows: int = None,
        ncols: int = None,
    ):
        """
        Executes the tool: validates input, processes data, and exports the result.

        Args:
            output_path: Directory to save the exported image.
            title: Optional title for the plot.
            figsize: Figure size for the plot.
            show_axis: Whether to display axes.
            colormap: Colormap for the image.
            show_colorbar: Whether to display a colorbar.
            filename_prefix: Prefix for the output filename.
            dpi: Dots per inch for the saved image.
            bbox_inches: Bounding box option for saving the figure.
            grid: Whether to display a grid.
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.

        Returns:
            self: The instance of the tool.
        """
        self._validate()
        self.process()
        self._export_file(
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
            # Forwarded rather than dropped: these were declared and documented
            # on execute() but never reached plt.subplots().
            1 if nrows is None else nrows,
            1 if ncols is None else ncols,
        )
        return self
