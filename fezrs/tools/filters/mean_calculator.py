# Import packages and libraries
from cv2 import blur

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


class MeanCalculator(BaseTool):
    def __init__(self, tif_path: BandPathType):
        super().__init__(tif_path=tif_path)

        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["tif"]
        )

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=["tif"]
        )

    def _validate(self):
        pass

    def process(self):
        self._output = blur(self.metadata_bands["tif"]["image_skimage"], (9, 9))
        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=1000,
        bbox_inches="tight",
        grid=False,
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
