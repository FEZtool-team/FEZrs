"""
process() then to_raster() is a documented second route (issue #66).

These tests build real GeoTIFFs and go through the public constructors, so
they catch the three defects the mocked per-tool tests could not: validation
that only ran inside execute(), a stacked file accepted as a band, and a
declared nodata fill treated as valid data.
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from fezrs import GeoeyeCalculator, NDVICalculator  # noqa: E402
from fezrs.tools.glcm.glcm_calculator import (  # noqa: E402
    DEFAULT_LEVELS,
    quantize_to_levels,
)
from fezrs.utils.file_handler import FileHandler  # noqa: E402

CRS = "EPSG:32639"
TRANSFORM = from_origin(316725.0, 4176795.0, 30.0, 30.0)
FILL = -9999


def _write(path, array, nodata=None):
    array = np.asarray(array)
    if array.ndim == 2:
        planes = array[np.newaxis, :, :]
    else:
        planes = array

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=planes.shape[1],
        width=planes.shape[2],
        count=planes.shape[0],
        dtype=array.dtype,
        crs=CRS,
        transform=TRANSFORM,
        nodata=nodata,
    ) as destination:
        destination.write(planes)
    return str(path)


def test_geoeye_process_rejects_a_single_band_raster(tmp_path):
    path = _write(tmp_path / "nir.tif", np.arange(16, dtype=np.int16).reshape(4, 4))

    with pytest.raises(ValueError, match="multi-band raster"):
        GeoeyeCalculator(tif_path=path, level=0).process()


def test_ndvi_rejects_a_stacked_geotiff_as_a_band(tmp_path):
    nir = np.array([[8000, 7000], [6000, 5000]], dtype=np.int16)
    red = np.array([[2000, 2500], [3000, 3500]], dtype=np.int16)
    stack = _write(tmp_path / "stack.tif", np.stack([nir, red]))

    with pytest.raises(ValueError, match="single-band raster"):
        NDVICalculator(nir_path=stack, red_path=stack)


def test_ndvi_over_declared_nodata_is_nan(tmp_path):
    nir = np.array([[8000, 7000], [FILL, 5000]], dtype=np.int16)
    red = np.array([[2000, 2500], [FILL, 3500]], dtype=np.int16)
    nir_path = _write(tmp_path / "nir.tif", nir, nodata=FILL)
    red_path = _write(tmp_path / "red.tif", red, nodata=FILL)

    result = NDVICalculator(nir_path=nir_path, red_path=red_path).process()

    assert np.isnan(result[1, 0])
    assert np.isfinite(result[0, 0])
    # Without masking, (-9999 - -9999) / (-9999 + -9999) is 0, which reads as
    # bare ground rather than as absence of data.
    assert result[1, 0] != 0


def test_glcm_quantization_is_not_dragged_by_nodata_fill(tmp_path):
    scene = np.arange(64, dtype=np.int16).reshape(8, 8)
    filled = scene.copy()
    filled[0, :] = FILL
    path = _write(tmp_path / "nir.tif", filled, nodata=FILL)

    loaded = FileHandler(nir_path=path).get_metadata_bands(["nir"])["nir"][
        "image_skimage"
    ]
    quantized = quantize_to_levels(loaded, DEFAULT_LEVELS)
    expected = quantize_to_levels(scene[1:, :].astype(float), DEFAULT_LEVELS)

    np.testing.assert_array_equal(quantized[1:, :], expected)


def test_to_raster_preserves_ndvi_nodata(tmp_path):
    nir = np.array([[8000, 7000], [FILL, 5000]], dtype=np.int16)
    red = np.array([[2000, 2500], [FILL, 3500]], dtype=np.int16)
    calculator = NDVICalculator(
        nir_path=_write(tmp_path / "nir.tif", nir, nodata=FILL),
        red_path=_write(tmp_path / "red.tif", red, nodata=FILL),
    )
    calculator.process()
    written = calculator.to_raster(tmp_path / "ndvi.tif")

    with rasterio.open(written) as source:
        array = source.read(1)
        assert np.isnan(array[1, 0])
        assert np.isfinite(array[0, 0])
