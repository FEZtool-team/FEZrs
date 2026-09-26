# Import packages and libraries
from cv2 import GaussianBlur

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.nodata_handler import apply_nodata, fill_invalid, invalid_mask
from fezrs.utils.type_handler import BandPathType


class GaussianCalculator(BaseTool):
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
            self._output = GaussianBlur(image, (13, 13), 0)
        else:
            # Mask the kernel footprint explicitly rather than rely on how
            # OpenCV happens to propagate NaN through a given filter.
            self._output = apply_nodata(
                GaussianBlur(fill_invalid(image, mask), (13, 13), 0), mask, window=13
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
