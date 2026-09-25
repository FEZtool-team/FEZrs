# Import packages and libraries
from cv2 import blur

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.nodata_handler import apply_nodata, fill_invalid, invalid_mask
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
        image = self.metadata_bands["tif"]["image_skimage"]
        mask = invalid_mask(image)

        if mask is None:
            self._output = blur(image, (9, 9))
        else:
            # cv2.blur is a running-sum box filter: once a NaN enters the sum it
            # never leaves, so everything to the right of the fill in each row
            # came back NaN. Filter filled data, then mask the 9x9 footprint.
            self._output = apply_nodata(
                blur(fill_invalid(image, mask), (9, 9)), mask, window=9
            )

        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix=None,
        dpi=1000,
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
