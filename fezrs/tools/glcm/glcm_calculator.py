import logging
import numpy as np
from pathlib import Path
from typing import Sequence, get_args
from fezrs.base import BaseTool
from skimage.feature import graycomatrix, graycoprops
from fezrs.utils.type_handler import BandPathType, PropertyGLCMType


logger = logging.getLogger(__name__)


# Haralick texture is direction dependent. Averaging the four principal
# orientations yields a rotation invariant measure, which is what a surface
# roughness or lithological discrimination map needs. A single angle is kept
# reachable for deliberately anisotropic work (bedding, foliation, lineaments).
DEFAULT_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_DISTANCES = (1,)

# Small windows supply very few co-occurrence pairs: a W x W window yields
# (W - 1) * W * 2 ordered pairs, i.e. 12 for W = 3. Spreading 12 pairs over a
# 256 x 256 matrix fills 0.018% of it, so the Haralick statistics end up
# describing matrix sparsity rather than surface texture. 32-64 levels is the
# usual working range for windowed GLCM.
DEFAULT_LEVELS = 64


def quantize_to_levels(image: np.ndarray, levels: int) -> np.ndarray:
    """
    Linearly quantize an image to ``levels`` gray levels in ``[0, levels - 1]``.

    Quantization is global: the scaling uses the whole-image minimum and
    maximum, so a texture value computed in one part of the scene is comparable
    with one computed elsewhere. That comparability is the point of a texture
    map, and a per-window rescale would destroy it.

    This replaces a plain ``astype(np.uint8)``, which wraps modulo 256 and
    destroys gray-level ordering on any raster holding values above 255 -- that
    is, on every 16-bit Landsat, Sentinel or ASTER band.

    Args:
        image: Input array of any numeric dtype.
        levels: Number of gray levels to quantize to.

    Returns:
        np.ndarray: ``uint8`` array with values in ``[0, levels - 1]``.
    """
    array = np.asarray(image, dtype=np.float64)

    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)

    low = array[finite].min()
    high = array[finite].max()

    # A constant band carries no texture; every pixel maps to level 0.
    if high == low:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array - low) / (high - low) * (levels - 1)
    scaled = np.where(finite, scaled, 0.0)

    return np.clip(np.round(scaled), 0, levels - 1).astype(np.uint8)


class GLCMCalculator(BaseTool):
    def __init__(
        self,
        nir_path: BandPathType,
        window_size: int = 3,
        propery: PropertyGLCMType | None = None,
        *,
        property: PropertyGLCMType | None = None,
        levels: int = DEFAULT_LEVELS,
        distances: Sequence[int] = DEFAULT_DISTANCES,
        angles: Sequence[float] = DEFAULT_ANGLES,
        centered: bool = True,
    ):
        """
        Args:
            nir_path: Single-band raster to compute texture on.
            window_size: Odd window edge length, >= 3.
            propery: Deprecated misspelling of ``property``, kept working.
            property: Haralick property to evaluate.
            levels: Gray levels to quantize to before building the GLCM.
            distances: Pixel offsets passed to ``graycomatrix``.
            angles: Orientations in radians; results are averaged over all of
                them, and over all distances.
            centered: Center the window on each pixel. When False, the window is
                anchored at the pixel and extends down and to the right, which
                offsets the texture map by ``(window_size - 1) // 2`` pixels
                relative to the source raster.
        """
        super().__init__(nir_path=nir_path)

        if propery is not None and property is not None:
            raise ValueError(
                "Pass either 'property' or the deprecated 'propery', not both."
            )

        selected_property = property if property is not None else propery
        self.property = "contrast" if selected_property is None else selected_property

        self.metadata_bands = self.files_handler.get_metadata_bands(
            requested_bands=["nir"]
        )

        self.result = np.empty(
            (self.metadata_bands["nir"]["height"], self.metadata_bands["nir"]["width"])
        )

        self.window_size = window_size
        self.levels = levels
        self.distances = tuple(distances)
        self.angles = tuple(angles)
        self.centered = centered

        self.nir_image = quantize_to_levels(
            self.metadata_bands["nir"]["image_skimage"], self.levels
        )

    def process(self):
        self._validate()

        height = self.metadata_bands["nir"]["height"]
        width = self.metadata_bands["nir"]["width"]

        offset = self.window_size // 2

        if self.centered:
            # Reflect padding keeps every window full size, so border pixels get
            # a texture value from a complete neighbourhood instead of from a
            # truncated one, and the window stays centered on its pixel.
            image = np.pad(self.nir_image, offset, mode="reflect")
        else:
            image = self.nir_image

        logger.debug(
            "GLCM %s over %dx%d, window=%d, levels=%d, distances=%s, angles=%s",
            self.property,
            height,
            width,
            self.window_size,
            self.levels,
            self.distances,
            self.angles,
        )

        for i in range(0, height):
            logger.debug("Processing row %d of %d", i, height)
            for j in range(0, width):
                window = image[i : i + self.window_size, j : j + self.window_size]

                glcm = graycomatrix(
                    window,
                    self.distances,
                    self.angles,
                    levels=self.levels,
                    normed=True,
                    symmetric=True,
                )
                # graycoprops returns (len(distances), len(angles)); averaging
                # over both gives the rotation invariant scalar.
                self.result[i, j] = graycoprops(glcm, self.property).mean()

        self._output = self.result
        return self._output

    def _validate(self):
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise ValueError("window_size must be an int.")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError(
                "window_size must be an odd integer greater than or equal to 3."
            )

        if not isinstance(self.levels, int) or isinstance(self.levels, bool):
            raise ValueError("levels must be an int.")
        if self.levels < 2 or self.levels > 256:
            raise ValueError("levels must be an int between 2 and 256.")

        if len(self.distances) == 0:
            raise ValueError("distances must contain at least one offset.")
        if any(
            not isinstance(distance, (int, np.integer))
            or isinstance(distance, bool)
            or distance < 1
            for distance in self.distances
        ):
            raise ValueError("distances must be positive integers.")

        if len(self.angles) == 0:
            raise ValueError("angles must contain at least one orientation.")

        valid_properties = get_args(PropertyGLCMType)
        if self.property not in valid_properties:
            raise ValueError(
                f"Invalid GLCM property: {self.property!r}. "
                f"Must be one of {list(valid_properties)}."
            )

        files_handler = getattr(self, "files_handler", None)
        if files_handler is None:
            return

        band_paths = getattr(files_handler, "band_paths", None)
        if not isinstance(band_paths, dict):
            return

        nir_path = band_paths.get("nir")
        if not nir_path or not Path(nir_path).is_file():
            raise FileNotFoundError(f"File {nir_path} not found")

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap=None,
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=500,
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
