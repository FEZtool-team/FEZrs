"""
Georeferenced raster export (issue #41).

Every tool except the mosaic calculator could only write a PNG rendered through
matplotlib: values quantized to 256 levels per channel, the pixel grid resampled
by dpi and bbox_inches, and CRS and transform discarded. The computed array was
correct and in memory, but the only documented output was a picture of it.
"""

from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine  # noqa: E402

from fezrs.base import BaseTool  # noqa: E402


CRS = "EPSG:32639"
# 30 m pixels at the origin of the bundled Landsat subset.
TRANSFORM = Affine(30.0, 0.0, 316725.0, 0.0, -30.0, 4176795.0)


class _Tool(BaseTool):
    """BaseTool with a stubbed handler, so nothing touches the filesystem."""

    def __init__(self, output, profile="default"):
        self._output = output
        self._BaseTool__tool_name = "Stub"

        if profile == "default":
            profile = {
                "crs": rasterio.crs.CRS.from_string(CRS),
                "transform": TRANSFORM,
                "nodata": None,
                "dtype": "int16",
                "height": 4,
                "width": 4,
            }

        handler = MagicMock()
        handler.get_raster_profile.return_value = profile
        self.files_handler = handler

    def _validate(self):
        pass

    def process(self):
        return self._output


CONTINUOUS = np.array(
    [
        [0.6237, -0.1042, 0.8891, 0.0],
        [0.3311, 0.7742, -0.5010, 0.1234],
        [0.9012, 0.0451, 0.2200, -0.9999],
        [0.1000, 0.5000, 0.7500, 0.3333],
    ]
)


def test_written_raster_carries_the_source_crs_and_transform(tmp_path):
    path = _Tool(CONTINUOUS).to_raster(tmp_path / "out.tif")

    with rasterio.open(path) as source:
        assert source.crs == rasterio.crs.CRS.from_string(CRS)
        assert source.transform == TRANSFORM


def test_values_survive_the_round_trip(tmp_path):
    """
    The PNG path quantized to 256 gray levels, so a pixel that was 0.6237 could
    not be recovered -- ruling out thresholding, zonal statistics and date
    differencing.
    """
    path = _Tool(CONTINUOUS).to_raster(tmp_path / "out.tif")

    with rasterio.open(path) as source:
        read = source.read(1)

    np.testing.assert_allclose(read, CONTINUOUS.astype("float32"))
    assert len(np.unique(read)) == CONTINUOUS.size


def test_pixel_grid_is_not_resampled(tmp_path):
    """The PNG was 9389x8035 for a 998x998 input, with no pixel correspondence."""
    path = _Tool(CONTINUOUS).to_raster(tmp_path / "out.tif")

    with rasterio.open(path) as source:
        assert source.shape == CONTINUOUS.shape


def test_continuous_output_defaults_to_float32_with_nan_nodata(tmp_path):
    path = _Tool(CONTINUOUS).to_raster(tmp_path / "out.tif")

    with rasterio.open(path) as source:
        assert source.dtypes[0] == "float32"
        assert np.isnan(source.nodata)


def test_integer_label_map_keeps_an_integer_dtype(tmp_path):
    """A classification is a label map; writing it as float with NaN is wrong."""
    labels = np.array([[1, 2], [2, 1]], dtype=int)

    path = _Tool(labels).to_raster(tmp_path / "classes.tif")

    with rasterio.open(path) as source:
        assert source.dtypes[0] == "int32"
        np.testing.assert_array_equal(source.read(1), labels)


def test_nan_survives_as_nodata(tmp_path):
    array = CONTINUOUS.copy()
    array[0, 0] = np.nan

    path = _Tool(array).to_raster(tmp_path / "out.tif")

    with rasterio.open(path) as source:
        read = source.read(1)
        assert np.isnan(read[0, 0])
        assert np.isnan(source.nodata)


def test_multi_component_output_becomes_a_multi_band_raster(tmp_path):
    """PCA returns (6, height, width)."""
    components = np.stack([CONTINUOUS + index for index in range(6)])

    path = _Tool(components).to_raster(tmp_path / "pca.tif")

    with rasterio.open(path) as source:
        assert source.count == 6
        assert source.shape == CONTINUOUS.shape
        for index in range(6):
            np.testing.assert_allclose(
                source.read(index + 1), components[index].astype("float32")
            )


def test_raster_is_tiled_and_compressed(tmp_path):
    """Scene-scale index rasters are large; these are the usual GeoTIFF options."""
    path = _Tool(CONTINUOUS).to_raster(tmp_path / "out.tif")

    with rasterio.open(path) as source:
        assert source.profile["tiled"] is True
        assert source.profile["compress"].lower() == "deflate"


def test_export_requires_a_computed_result(tmp_path):
    with pytest.raises(ValueError, match="Data not computed"):
        _Tool(None).to_raster(tmp_path / "out.tif")


def test_missing_crs_raises_instead_of_writing_an_identity_transform(tmp_path):
    """
    Silently writing an identity transform would produce a file that looks
    georeferenced and places the scene at the origin.
    """
    profile = {
        "crs": None,
        "transform": TRANSFORM,
        "nodata": None,
        "dtype": "int16",
        "height": 4,
        "width": 4,
    }

    with pytest.raises(ValueError, match="carries no CRS"):
        _Tool(CONTINUOUS, profile=profile).to_raster(tmp_path / "out.tif")


def test_parent_directories_are_created(tmp_path):
    path = _Tool(CONTINUOUS).to_raster(tmp_path / "nested" / "deeper" / "out.tif")

    assert Path(path).is_file()


# --- Against the bundled scene ------------------------------------------------


EXAMPLE_DATA = Path(__file__).resolve().parent.parent / "example" / "data"


@pytest.mark.skipif(
    not (EXAMPLE_DATA / "nir.tif").is_file(), reason="bundled example data absent"
)
def test_end_to_end_against_the_bundled_scene(tmp_path):
    from fezrs import NDVICalculator

    calculator = NDVICalculator(
        nir_path=EXAMPLE_DATA / "nir.tif", red_path=EXAMPLE_DATA / "red.tif"
    )
    computed = calculator.process()
    path = calculator.to_raster(tmp_path / "ndvi.tif")

    with rasterio.open(EXAMPLE_DATA / "nir.tif") as source, rasterio.open(path) as out:
        assert out.crs == source.crs
        assert out.transform == source.transform
        assert out.shape == source.shape
        np.testing.assert_allclose(out.read(1), computed.astype("float32"))


@pytest.mark.skipif(
    not (EXAMPLE_DATA / "nir.tif").is_file(), reason="bundled example data absent"
)
def test_file_handler_reads_the_source_profile():
    from fezrs.utils.file_handler import FileHandler

    handler = FileHandler(nir_path=str(EXAMPLE_DATA / "nir.tif"))
    profile = handler.get_raster_profile()

    assert profile["crs"].to_string() == CRS
    assert profile["transform"] == TRANSFORM
    assert (profile["height"], profile["width"]) == (998, 998)


def test_raster_profile_is_none_without_a_source():
    from fezrs.utils.file_handler import FileHandler

    assert FileHandler().get_raster_profile() is None
