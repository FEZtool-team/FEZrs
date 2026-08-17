from uuid import uuid4
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class HistogramExportMixin:
    """
    Mixin class providing methods to add a watermark and save histogram figures.

    Intended to be used with classes that have a '_logo_watermark' attribute.
    """

    def _add_watermark(self, ax):
        """
        Adds a semi-transparent watermark logo to the given matplotlib axes.

        Args:
            ax: The matplotlib axes object to which the watermark will be added.
        """
        imagebox = OffsetImage(self._logo_watermark, zoom=1, alpha=0.3)
        ab = AnnotationBbox(
            imagebox,
            (0.95, 0.95),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(1, 1),
        )
        ax.add_artist(ab)

    def _save_histogram_figure(
        self, ax, output_path, filename_prefix, dpi, bbox_inches
    ):
        """
        Saves the histogram figure to a PNG file and closes the figure.

        Args:
            ax: The matplotlib axes object containing the histogram.
            output_path: Directory to save the exported image.
            filename_prefix: Prefix for the output filename.
            dpi: Dots per inch for the saved image.
            bbox_inches: Bounding box option for saving the figure.

        Returns:
            The path to the saved image file.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{output_path}/{filename_prefix}_{uuid4().hex}.png"
        ax.figure.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(ax.figure)
        return filename
