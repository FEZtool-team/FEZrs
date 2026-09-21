import pytest
from unittest import mock
import numpy as np

from fezrs.utils.file_handler import (
    FileHandler,
    _as_single_band,
    _load_image,
    _mask_nodata,
    _normalize,
)


def test_load_image_none_path():
    assert _load_image(None) is None


@mock.patch("fezrs.utils.file_handler.os.path.exists", return_value=True)
@mock.patch(
    "fezrs.utils.file_handler.io.imread", return_value=np.array([[1, 2], [3, 4]])
)
def test_load_image_valid_path(mock_imread, mock_exists):
    result = _load_image("image.tif")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]], dtype=float))


@mock.patch("fezrs.utils.file_handler.os.path.exists", return_value=False)
def test_load_image_file_not_found(mock_exists):
    with pytest.raises(FileNotFoundError):
        _load_image("nonexistent.jpg")


def test_as_single_band_accepts_2d():
    array = np.arange(6, dtype=float).reshape(2, 3)

    result = _as_single_band(array, "nir", "nir.tif")

    np.testing.assert_array_equal(result, array)


def test_as_single_band_squeezes_singleton_dimension():
    array = np.arange(6, dtype=float).reshape(2, 3, 1)

    result = _as_single_band(array, "red", "red.tif")

    assert result.shape == (2, 3)


def test_as_single_band_rejects_stacked_raster():
    array = np.arange(12, dtype=float).reshape(2, 3, 2)

    with pytest.raises(ValueError, match="single-band raster"):
        _as_single_band(array, "nir", "stack.tif")


def test_normalize_ignores_nan_when_scaling():
    image = np.array([[0.0, 10.0], [np.nan, 5.0]])

    result = _normalize(image)

    np.testing.assert_allclose(result[0], [0.0, 1.0])
    assert np.isnan(result[1, 0])
    np.testing.assert_allclose(result[1, 1], 0.5)


def test_normalize_all_nan_stays_nan():
    image = np.full((2, 2), np.nan)

    result = _normalize(image)

    assert np.isnan(result).all()


def _write_tif(path, array, nodata=None, count=None):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    array = np.asarray(array)
    if count is None:
        if array.ndim == 2:
            count = 1
            planes = array[np.newaxis, :, :]
        else:
            count = array.shape[0]
            planes = array

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=planes.shape[1],
        width=planes.shape[2],
        count=count,
        dtype=planes.dtype,
        crs="EPSG:32639",
        transform=from_origin(316725.0, 4176795.0, 30.0, 30.0),
        nodata=nodata,
    ) as destination:
        destination.write(planes)
    return str(path)


def test_file_handler_rejects_stacked_named_band(tmp_path):
    stack = np.stack(
        [
            np.arange(9, dtype=np.int16).reshape(3, 3),
            np.arange(9, 18, dtype=np.int16).reshape(3, 3),
        ]
    )
    path = _write_tif(tmp_path / "stack.tif", stack)

    with pytest.raises(ValueError, match="nir_path must be a single-band raster"):
        FileHandler(nir_path=path)


def test_file_handler_masks_declared_nodata(tmp_path):
    fill = -9999
    values = np.array([[100, 200], [fill, 300]], dtype=np.int16)
    path = _write_tif(tmp_path / "nir.tif", values, nodata=fill)

    handler = FileHandler(nir_path=path)

    loaded = handler.bands["nir"]
    np.testing.assert_array_equal(loaded[0], [100.0, 200.0])
    assert np.isnan(loaded[1, 0])
    assert loaded[1, 1] == 300.0


def test_mask_nodata_leaves_array_unchanged_without_profile():
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = _mask_nodata(array, "not-a-real-file.tif")

    np.testing.assert_array_equal(result, array)


def test_metadata_bands_mask_nodata(tmp_path):
    fill = -9999
    values = np.array([[10, fill], [20, 30]], dtype=np.int16)
    path = _write_tif(tmp_path / "nir.tif", values, nodata=fill)

    metadata = FileHandler(nir_path=path).get_metadata_bands(["nir"])
    image = metadata["nir"]["image_skimage"]

    assert np.isnan(image[0, 1])
    assert image[0, 0] == 10.0
