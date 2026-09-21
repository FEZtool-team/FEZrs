import os
import numpy as np
from skimage import io
import rasterio as rio
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from fezrs.utils.type_handler import BandPathType, BandNameType, BandTypes


def _mask_nodata(array: np.ndarray, path: BandPathType) -> np.ndarray:
    """
    Replace a raster's declared nodata fill with NaN.

    ``skimage.io.imread`` (and ``plt.imread``) return the fill as ordinary
    values. A Landsat/Sentinel margin of -9999 then participates in ratios
    (NDVI of the fill is 0) and in global min/max (GLCM quantization spends
    most of its gray levels on the gap between fill and data).

    Rasterio is used only to read the profile; if that fails the array is
    returned unchanged so non-georeferenced inputs keep working.
    """
    try:
        nodata = _raster_profile(str(path)).get("nodata")
    except Exception:
        return array

    if nodata is None:
        return array

    masked = np.asarray(array, dtype=float)
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        return masked

    return np.where(masked == nodata, np.nan, masked)


def _as_single_band(
    array: np.ndarray, band_name: str, path: BandPathType
) -> np.ndarray:
    """
    Require a named band path to be a 2-D array.

    A stacked multi-band GeoTIFF handed to ``nir_path`` / ``red_path`` is a
    common mistake with a stacked delivery. Arithmetic then broadcasts across
    the extra axis and writes a plausible-looking, wrong product.

    A trailing or leading singleton dimension is squeezed: some readers return
    ``(height, width, 1)`` for a single-band file.
    """
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return squeezed

    raise ValueError(
        f"{band_name}_path must be a single-band raster, but {path} has "
        f"shape {array.shape}. Pass a 2-D band, not a stacked multi-band "
        "GeoTIFF."
    )


def _load_image(path: Optional[BandPathType]) -> Optional[np.ndarray]:
    """
    Loads an image from the specified file path if it exists.

    Args:
        path (Optional[str]): The file path to the image. If None, the function returns None.

    Returns:
        Optional[np.ndarray]: The loaded image as a NumPy array with float type, or None if the path is None.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    # TODO - Add a check for file type, files must be in (*.tiff | *.tif) format

    if path and os.path.exists(path):
        return _mask_nodata(io.imread(path).astype(float), path)
    elif path is None:
        return None
    else:
        raise FileNotFoundError(f"File {path} not found")


def _normalize(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalize a given image array to the range [0, 1].

    Nodata (NaN) is ignored when computing the extrema so a fill value cannot
    collapse the stretch, and is preserved in the result.

    Args:
        image (Optional[np.ndarray]): The input image as a NumPy array.
            If None, the function returns None.

    Returns:
        Optional[np.ndarray]: The normalized image array with values scaled
            to the range [0, 1], or None if the input is None.

    Raises:
        TypeError: If the input is not a NumPy array.
    """
    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, but got {type(image)}")

    finite = np.isfinite(image)
    if not finite.any():
        return np.full(image.shape, np.nan, dtype=float)

    low = np.min(image[finite])
    high = np.max(image[finite])
    if high == low:
        return np.where(finite, 0.0, np.nan)

    return np.where(finite, (image - low) / (high - low), np.nan)


def _metadata_image(path: str) -> Dict[str, np.ndarray]:
    """
    Extracts metadata for a given image file.

    This function reads an image from the specified file path using both Matplotlib
    and scikit-image libraries. It returns a dictionary containing the image data
    from both libraries, as well as the image's height and width.

    Args:
        path (str): The file path to the image.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - "image_plt": The image data read using Matplotlib.
            - "image_skimage": The image data read using scikit-image.
            - "height": The height of the image (number of rows).
            - "width": The width of the image (number of columns).
    """
    image_plt = _mask_nodata(plt.imread(path), path)
    image_skimage = _mask_nodata(io.imread(path), path)
    return {
        "image_plt": image_plt,
        "image_skimage": image_skimage,
        "height": image_plt.shape[0],
        "width": image_plt.shape[1],
    }


def _rasterio_image_tifs(path: str):
    """
    Opens a TIFF image using rasterio.

    Args:
        path (str): The file path to the TIFF image.

    Returns:
        rasterio.io.DatasetReader: The rasterio dataset object for the image.
    """
    return rio.open(path)


def _raster_profile(path: str) -> Dict:
    """
    Read the spatial referencing of a raster without loading its pixels.

    Args:
        path (str): Path to a raster file.

    Returns:
        Dict: CRS, affine transform, nodata value, dtype and shape.
    """
    with rio.open(path) as source:
        return {
            "crs": source.crs,
            "transform": source.transform,
            "nodata": source.nodata,
            "dtype": source.dtypes[0],
            "height": source.height,
            "width": source.width,
        }


class FileHandler:
    """
    FileHandler is a utility class for managing and processing geospatial image files.

    It provides functionality to load, normalize, and retrieve metadata for various image bands.

    Attributes:
        tif_paths (Optional[List[BandPathType]]):
            List of file paths for multi-band TIFF images.
        band_paths (Dict[str, Optional[BandPathType]]):
            A dictionary mapping band names (e.g., "red", "nir") to their respective file paths.
        bands (Dict[str, Optional[np.ndarray]]):
            A dictionary mapping band names to their loaded image data as NumPy arrays.

    Methods:
        get_normalized_bands(requested_bands: Optional[List[BandNameType]] = None) -> Dict[str, Optional[np.ndarray]]:
            Retrieve normalized versions of the requested image bands. If no bands are specified, all available bands are normalized.

        get_metadata_bands(requested_bands: Optional[List[BandNameType]] = None) -> Dict[str, Dict]:
            Retrieve metadata (image data and dimensions) for the requested image bands. If no bands are specified, metadata for all available bands is returned.

        get_images_collection() -> list:
            Retrieve the loaded arrays for every supplied band, in band_paths order.

        get_rasterio_tifs(requested_bands: Optional[List[BandNameType]] = None):
            Retrieve rasterio objects for all TIFF paths in tif_paths. Raises ValueError if tif_paths is None.
    """

    def __init__(
        self,
        red_path: Optional[BandPathType] = None,
        green_path: Optional[BandPathType] = None,
        blue_path: Optional[BandPathType] = None,
        nir_path: Optional[BandPathType] = None,
        swir1_path: Optional[BandPathType] = None,
        swir2_path: Optional[BandPathType] = None,
        tif_path: Optional[BandPathType] = None,
        # Tif list bands path
        tif_paths: Optional[List[BandPathType]] = None,
        # Before bands paths
        before_nir_path: Optional[BandPathType] = None,
        before_swir1_path: Optional[BandPathType] = None,
        before_swir2_path: Optional[BandPathType] = None,
    ):
        """
        Initialize the FileHandler with paths to various image bands.

        Args:
            red_path (Optional[BandPathType]): Path to the red band image.
            green_path (Optional[BandPathType]): Path to the green band image.
            blue_path (Optional[BandPathType]): Path to the blue band image.
            nir_path (Optional[BandPathType]): Path to the near-infrared band image.
            swir1_path (Optional[BandPathType]): Path to the shortwave infrared 1 band image.
            swir2_path (Optional[BandPathType]): Path to the shortwave infrared 2 band image.
            tif_path (Optional[BandPathType]): Path to a single TIFF image.
            tif_paths (Optional[List[BandPathType]]): List of paths to TIFF images.
            before_nir_path (Optional[BandPathType]): Path to the "before" NIR band image.
            before_swir1_path (Optional[BandPathType]): Path to the "before" SWIR1 band image.
            before_swir2_path (Optional[BandPathType]): Path to the "before" SWIR2 band image.
        """
        self.tif_paths = tif_paths

        self.band_paths: BandTypes = {
            "tif": tif_path,
            "red": red_path,
            "nir": nir_path,
            "blue": blue_path,
            "swir1": swir1_path,
            "swir2": swir2_path,
            "green": green_path,
            "before_nir": before_nir_path,
            "before_swir1": before_swir1_path,
            "before_swir2": before_swir2_path,
        }

        self.bands: BandTypes = {
            key: _load_image(path) for key, path in self.band_paths.items()
        }

        # Named band paths are one plane. ``tif`` is the exception: Geoeye
        # (and similar import tools) index a stacked multi-band raster.
        for key, array in self.bands.items():
            if array is None or key == "tif":
                continue
            self.bands[key] = _as_single_band(array, key, self.band_paths[key])

    def get_normalized_bands(
        self, requested_bands: Optional[List[BandNameType]] = None
    ):
        """
        Retrieve normalized versions of the requested image bands.

        Args:
            requested_bands (Optional[List[BandNameType]]): A list of band names to normalize.
                If None, all available bands will be normalized.

        Returns:
            Dict[str, Optional[np.ndarray]]: A dictionary mapping band names to their normalized image data.
                Bands with no data will be excluded from the result.
        """
        if requested_bands is None:
            requested_bands = list(self.bands.keys())

        return {
            band: _normalize(self.bands[band])
            for band in requested_bands
            if self.bands.get(band) is not None
        }

    def get_bands(self, requested_bands: Optional[List[BandNameType]] = None):
        """
        Retrieve the requested image bands with their values as read.

        Unlike :meth:`get_normalized_bands`, no rescaling is applied. Spectral
        indices must use this accessor: a per-band min-max rescale gives each
        band a different affine transform, which alters the relationships
        *between* bands, and those relationships are the entire physical content
        of a band ratio.

        Args:
            requested_bands (Optional[List[BandNameType]]): A list of band names
                to return. If None, all available bands are returned.

        Returns:
            Dict[str, Optional[np.ndarray]]: A dictionary mapping band names to
                their image data as read. Bands with no data are excluded.
        """
        if requested_bands is None:
            requested_bands = list(self.bands.keys())

        return {
            band: self.bands[band]
            for band in requested_bands
            if self.bands.get(band) is not None
        }

    def get_raster_profile(
        self, band: Optional[BandNameType] = None
    ) -> Optional[Dict]:
        """
        Retrieve the spatial referencing of an input band.

        Bands are loaded for computation through scikit-image, which discards
        CRS, transform and nodata. This reads that metadata back via rasterio so
        a result can be written out as a georeferenced raster rather than only
        as a picture of one.

        Args:
            band (Optional[BandNameType]): Band to describe. Defaults to the
                first band that was supplied.

        Returns:
            Optional[Dict]: Profile mapping, or None when no source is available.
        """
        if band is None:
            candidates = [
                name
                for name, path in self.band_paths.items()
                if path is not None
            ]
            if not candidates and self.tif_paths:
                return _raster_profile(str(self.tif_paths[0]))
            if not candidates:
                return None
            band = candidates[0]

        path = self.band_paths.get(band)
        if path is None or not os.path.exists(path):
            return None

        return _raster_profile(str(path))

    def get_metadata_bands(
        self, requested_bands: Optional[list[BandNameType]] = None
    ) -> Dict[str, Dict]:
        """
        Retrieve metadata for the requested image bands.

        Args:
            requested_bands (Optional[List[BandNameType]]): A list of band names to retrieve metadata for.
                If None, metadata for all available bands will be retrieved.

        Returns:
            Dict[str, Dict]: A dictionary mapping band names to their metadata.
                Metadata includes image data and dimensions (height and width).
        """
        if requested_bands is None:
            requested_bands = self.bands.keys()

        metadata = {}
        for band in requested_bands:
            path = self.band_paths.get(band)
            if path and os.path.exists(path):
                metadata[band] = _metadata_image(path)
                if band != "tif":
                    # ``image_plt`` can be RGB even for a single-band TIFF;
                    # tools compute on ``image_skimage``.
                    metadata[band]["image_skimage"] = _as_single_band(
                        metadata[band]["image_skimage"], band, path
                    )
                    metadata[band]["height"] = int(
                        metadata[band]["image_skimage"].shape[0]
                    )
                    metadata[band]["width"] = int(
                        metadata[band]["image_skimage"].shape[1]
                    )

        return metadata

    def get_images_collection(self) -> any:
        """
        Retrieve the loaded arrays for every supplied band.

        Returns:
            list[np.ndarray]: Band arrays in ``band_paths`` insertion order,
            already float-converted and nodata-masked.
        """
        # Reuse the arrays already loaded into ``self.bands`` so PCA/SVM see
        # the same nodata-masked floats as the rest of the library. Returning
        # a list (not ImageCollection) is enough: callers only iterate and
        # concatenate. Order stays the ``band_paths`` insertion order.
        return [
            self.bands[key]
            for key, path in self.band_paths.items()
            if path is not None
        ]

    def get_rasterio_tifs(self, requested_bands: Optional[list[BandNameType]] = None):
        """
        Retrieve rasterio DatasetReader objects for all TIFF paths in tif_paths.

        Args:
            requested_bands (Optional[List[BandNameType]]): Not used. Reserved for future filtering.

        Returns:
            List[rasterio.io.DatasetReader]: List of rasterio dataset objects for each TIFF path.

        Raises:
            ValueError: If tif_paths is None.
        """
        if self.tif_paths is None:
            raise ValueError("The <tif_paths> could not be empty to read by rasterio.")

        rasterio_image = []
        for tif_path in self.tif_paths:
            path = tif_path
            if path and os.path.exists(path):
                rasterio_image.append(_rasterio_image_tifs(path))

        return rasterio_image
