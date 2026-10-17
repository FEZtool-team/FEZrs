# Import packages and libraries
import numpy as np
from cv2 import medianBlur
from skimage.filters import median as skimage_median

# Import module and files
from fezrs.base import BaseTool
from fezrs.utils.nodata_handler import apply_nodata, fill_invalid, invalid_mask
from fezrs.utils.type_handler import BandPathType


# OpenCV's median filter only accepts some dtype / kernel combinations: any
# kernel on uint8, but only 3x3 and 5x5 on uint16, int16 and float32, and nothing
# at all on float64. A 7x7 median on 16-bit Landsat or Sentinel data therefore
# raised cv2.error, as does any band carrying a nodata fill (masked to float).
_CV2_SMALL_KERNEL_DTYPES = (np.uint16, np.int16, np.float32)


def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Median filter for any dtype and any odd kernel size.

    Uses OpenCV where it supports the combination, which keeps the established
    output bit-identical, and scikit-image's exact median otherwise. Both
    replicate the edge pixels at the border, so they agree there too.
    """
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return medianBlur(array, ksize=kernel_size)
    if kernel_size <= 5 and array.dtype.type in _CV2_SMALL_KERNEL_DTYPES:
        return medianBlur(array, ksize=kernel_size)
    if kernel_size <= 5 and np.issubdtype(array.dtype, np.floating):
        # float64 -> float32 is exact for any 16-bit digital number.
        return medianBlur(array.astype(np.float32), ksize=kernel_size)

    footprint = np.ones((kernel_size, kernel_size), dtype=bool)
    return skimage_median(array, footprint=footprint, mode="nearest")


class MedianCalculator(BaseTool):
    def __init__(self, tif_path: BandPathType, kernel_size: int):
        super().__init__(tif_path=tif_path)

        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["tif"]
        )

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=["tif"]
        )

        self.kernel_size = kernel_size

    def _validate(self):
        # Validate kernel_size
        if not isinstance(self.kernel_size, int):
            raise TypeError(
                f"'kernel_size' must be an integer, got {type(self.kernel_size).__name__}"
            )
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("'kernel_size' must be a positive odd integer")

        # Validate tif_band
        tif_band = self.files_handler.bands.get("tif")
        if tif_band is None:
            raise ValueError("No 'tif' band found in files_handler.bands")
        if not isinstance(tif_band, np.ndarray):
            raise TypeError(
                f"'tif' band must be a NumPy ndarray, got {type(tif_band).__name__}"
            )
        if tif_band.ndim != 2:
            raise ValueError("'tif' band must be a 2D array")

        # Validate metadata
        metadata = self.metadata_bands.get("tif")
        if not metadata:
            raise ValueError("Missing metadata for 'tif' band")
        if not isinstance(metadata.get("width"), int) or metadata["width"] <= 0:
            raise ValueError("Invalid 'width' in tif metadata")
        if not isinstance(metadata.get("height"), int) or metadata["height"] <= 0:
            raise ValueError("Invalid 'height' in tif metadata")

    def process(self):
        image = self.metadata_bands["tif"]["image_skimage"]
        mask = invalid_mask(image)

        if mask is None:
            self._output = median_filter(image, self.kernel_size)
        else:
            # A median that reaches into the fill is not a measurement, so any
            # pixel whose window touches nodata is reported as NaN.
            filtered = median_filter(fill_invalid(image, mask), self.kernel_size)
            self._output = apply_nodata(filtered, mask, window=self.kernel_size)

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
