# Import packages and libraries
from abc import ABC
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
        filename_prefix: str = "Tool_output",
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

    def execute(
        self,
        output_path: BandPathType,
        title: str | None = None,
        figsize: tuple = (10, 10),
        show_axis: bool = False,
        colormap: str = None,
        show_colorbar: bool = False,
        filename_prefix: str = "Tool_output",
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
        )
        return self
