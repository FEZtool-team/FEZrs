"""
Declared nodata, across every calculator (issue #66).

Landsat Collection 2 and Sentinel-2 deliveries routinely carry a fill margin with
its value declared in the GeoTIFF profile. Read as ordinary data, that fill
reports an NDVI of 0 (bare ground or water rather than "not observed"), claims
the ground was unburned, becomes a cluster of its own, and -- because GLCM
quantizes on global extrema -- spends most of the gray-level range on the gap
between fill and data.

Masking the fill to NaN on read fixes the silent cases, but many of the wrapped
algorithms cannot take NaN, so on its own that trades wrong answers for crashes.
These tests pin the whole contract: every calculator runs on nodata-declared
input, reports the fill as NaN, and keeps its valid pixels.
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

import fezrs  # noqa: E402
from tests.smoke_test import (  # noqa: E402
    CASE_NAMES,
    DPI,
    GLCM_SIZE,
    SIZE,
    _band,
    _cases,
    _kwargs_for,
)

FILL = -9999
CRS = "EPSG:32639"
TRANSFORM = from_origin(316725.0, 4176795.0, 30.0, 30.0)

# A vertical gap mid-scene, like a Landsat 7 SLC-off stripe. Kept clear of the
# SVM training samples the smoke cases place near the corners.
STRIPE = slice(8, 12)
GLCM_STRIPE = slice(2, 4)


def _with_fill(array, stripe=STRIPE):
    filled = array.copy()
    filled[:, stripe] = FILL
    return filled


def _write(path, array, nodata=FILL):
    count = 1 if array.ndim == 2 else array.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[-2],
        width=array.shape[-1],
        count=count,
        dtype=array.dtype,
        crs=CRS,
        transform=TRANSFORM,
        nodata=nodata,
    ) as destination:
        if array.ndim == 2:
            destination.write(array, 1)
        else:
            destination.write(array)
    return str(path)


@pytest.fixture(scope="session")
def nodata_bands(tmp_path_factory):
    """Every smoke-test input, with a declared -9999 fill stripe."""
    directory = tmp_path_factory.mktemp("nodata_bands")

    spec = {
        "blue": (1, 300, 900),
        "green": (2, 400, 1200),
        "red": (3, 350, 1400),
        "nir": (4, 1800, 5200),
        "swir1": (5, 900, 3000),
        "swir2": (6, 600, 2200),
        "before_nir": (7, 1500, 4700),
        "before_swir1": (8, 1000, 3300),
        "before_swir2": (9, 700, 2500),
        "tif": (10, 200, 6000),
    }
    paths = {
        name: _write(directory / f"{name}.tif", _with_fill(_band(seed, low, high)))
        for name, (seed, low, high) in spec.items()
    }
    paths["glcm"] = _write(
        directory / "glcm.tif",
        _with_fill(_band(11, 100, 4000, size=GLCM_SIZE), GLCM_STRIPE),
    )
    paths["multiband"] = _write(
        directory / "multiband.tif",
        np.stack([_with_fill(_band(30 + i, 200, 3000)) for i in range(4)]),
    )

    for index, offset in enumerate((0.0, SIZE * 15.0)):
        path = directory / f"tile_{index}.tif"
        tile = _with_fill(_band(20 + index, 300, 3000))
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=SIZE,
            width=SIZE,
            count=1,
            dtype=tile.dtype,
            crs=CRS,
            transform=from_origin(316725.0 + offset, 4176795.0, 30.0, 30.0),
            nodata=FILL,
        ) as destination:
            destination.write(tile, 1)
        paths[f"tile_{index}"] = str(path)

    return paths


def _build(name, bands):
    return getattr(fezrs, name)(**_kwargs_for(name, bands))


def _as_bands(output, grid):
    """Output as (bands, height, width), whatever its native axis order."""
    array = np.asarray(output, dtype=float)
    if array.ndim == 2:
        return array[np.newaxis]
    if array.shape[1:] == grid:
        return array
    return np.moveaxis(array, -1, 0)


# --- Every calculator ---------------------------------------------------------


@pytest.mark.parametrize("name", CASE_NAMES)
def test_runs_on_nodata_input(name, nodata_bands, tmp_path):
    """
    execute(), histogram_export() and to_raster() all succeed. Before these
    fixes five calculators raised here: OpenCV's median filter rejects float64,
    scikit-image cannot histogram NaN, and KMeans and SVC refuse missing values.
    """
    calculator = _build(name, nodata_bands)

    output = tmp_path / name
    calculator.execute(output_path=output, dpi=DPI)
    assert list(output.glob("*.png")), f"{name}.execute() wrote nothing"

    if "histogram_export" in vars(type(calculator)):
        histogram = tmp_path / f"{name}_hist"
        histogram.mkdir()
        _build(name, nodata_bands).histogram_export(output_path=histogram, dpi=DPI)
        assert list(histogram.glob("*.png"))

    if name != "MosaicCalculator":
        grid = (GLCM_SIZE, GLCM_SIZE) if name == "GLCMCalculator" else (SIZE, SIZE)
        fresh = _build(name, nodata_bands)
        fresh.process()
        fresh.to_raster(tmp_path / f"{name}.tif")
        with rasterio.open(tmp_path / f"{name}.tif") as source:
            assert (source.height, source.width) == grid


@pytest.mark.parametrize("name", CASE_NAMES)
def test_fill_is_reported_as_nan(name, nodata_bands):
    """
    The fill comes back as NaN -- never as a number that reads like a
    measurement -- while valid data survives.
    """
    if name == "MosaicCalculator":
        pytest.skip("MosaicCalculator's _output is a file path, not an array")

    calculator = _build(name, nodata_bands)
    calculator.process()

    grid = (GLCM_SIZE, GLCM_SIZE) if name == "GLCMCalculator" else (SIZE, SIZE)
    stripe = GLCM_STRIPE if name == "GLCMCalculator" else STRIPE
    bands = _as_bands(calculator._output, grid)

    assert np.all(np.isnan(bands[:, :, stripe])), (
        f"{name} reported values over the declared nodata stripe"
    )
    assert np.isfinite(bands[:, :, -1]).any(), (
        f"{name} lost the valid data along with the fill"
    )


# --- The reviewers' specific cases --------------------------------------------


def test_ndvi_over_fill_is_nan_not_zero(tmp_path):
    """
    (-9999 - -9999) / (-9999 + -9999) is 0, so the fill read as NDVI 0 -- bare
    ground or water -- rather than as absence of data. The valid region must be
    unaffected: NIR 3000 over red 1000 is 0.5.
    """
    nir = np.full((6, 8), 3000, dtype=np.int16)
    red = np.full((6, 8), 1000, dtype=np.int16)
    nir[:, :3] = FILL
    red[:, :3] = FILL

    ndvi = fezrs.NDVICalculator(
        nir_path=_write(tmp_path / "nir.tif", nir),
        red_path=_write(tmp_path / "red.tif", red),
    ).process()

    assert np.all(np.isnan(ndvi[:, :3]))
    np.testing.assert_allclose(ndvi[:, 3:], 0.5)


def test_glcm_quantization_is_not_spent_on_the_fill(tmp_path):
    """
    Global min/max quantization is right, but with the fill included it set
    one end of the range: the reviewer measured 25 of 64 gray levels in use on
    the valid part of the scene, compressing exactly the texture contrast that
    separates lithological units.

    The exact check: adding a fill margin must not change how the valid pixels
    quantize at all.
    """
    from fezrs.tools.glcm.glcm_calculator import quantize_to_levels

    valid = _band(40, 200, 4000, size=16)
    with_margin = np.hstack([np.full((16, 4), FILL, dtype=valid.dtype), valid])

    calculator = fezrs.GLCMCalculator(
        nir_path=_write(tmp_path / "glcm.tif", with_margin), window_size=3, levels=64
    )

    np.testing.assert_array_equal(
        calculator.nir_image[:, 4:], quantize_to_levels(valid, 64)
    )

    # And what the old behaviour did to the same pixels, for the record.
    fill_in_range = quantize_to_levels(with_margin.astype(float), 64)[:, 4:]
    assert np.unique(fill_in_range).size < np.unique(calculator.nir_image[:, 4:]).size


def test_glcm_does_not_invent_a_texture_edge_at_the_fill(tmp_path):
    """
    A window reaching into the fill measures the contrast between data and the
    fill level. Along a scene edge that reads as a sharp texture boundary,
    indistinguishable from a lithological contact.
    """
    band = _band(41, 1000, 1200, size=12)  # smooth: no real texture edge
    band[:, :4] = FILL

    texture = fezrs.GLCMCalculator(
        nir_path=_write(tmp_path / "glcm.tif", band), window_size=3
    ).process()

    # Columns 0-3 are fill; column 4's window reaches column 3.
    assert np.all(np.isnan(texture[:, :5]))
    assert np.all(np.isfinite(texture[:, 5:]))


def test_kmeans_spends_every_cluster_on_valid_data(nodata_bands):
    """The fill, being one repeated value, would otherwise take a whole cluster."""
    output = fezrs.KMeansCalculator(
        nir_path=nodata_bands["nir"], n_clusters=3, random_state=0
    ).process()

    finite = output[np.isfinite(output)]
    assert np.unique(finite).size == 3


def test_svm_rejects_a_training_sample_on_nodata(nodata_bands):
    kwargs = dict(_kwargs_for("SVMCalculator", nodata_bands))
    kwargs["training_samples"] = [(5, 9, 1), (20, 20, 2)]  # column 9 is fill

    with pytest.raises(ValueError, match="falls on nodata"):
        fezrs.SVMCalculator(**kwargs).process()


def test_svm_leaves_the_fill_unclassified(nodata_bands):
    """A fill pixel given a class would read as land cover."""
    output = fezrs.SVMCalculator(**_kwargs_for("SVMCalculator", nodata_bands)).process()

    assert np.all(np.isnan(output[:, STRIPE]))
    classes = set(np.unique(output[np.isfinite(output)]).tolist())
    assert classes <= {1.0, 2.0} and classes


def test_burn_map_marks_the_fill_as_unobserved(nodata_bands):
    """
    A boolean map cannot hold "no data", so the fill compared as False and was
    reported as unburned -- a confident claim about ground nobody observed.
    """
    output = fezrs.BurnCalculator(**_kwargs_for("BurnCalculator", nodata_bands)).process()

    assert np.all(np.isnan(output[:, STRIPE]))
    assert set(np.unique(output[np.isfinite(output)]).tolist()) <= {0.0, 1.0}


def test_burn_map_without_nodata_is_still_boolean(tmp_path):
    """Inputs that declare no nodata keep exactly their previous output."""
    bands = {
        name: _write(tmp_path / f"{name}.tif", _band(seed, 500, 5000), nodata=None)
        for name, seed in (("nir", 1), ("swir2", 2), ("before_nir", 3), ("before_swir2", 4))
    }
    output = fezrs.BurnCalculator(
        nir_path=bands["nir"],
        swir2_path=bands["swir2"],
        before_nir_path=bands["before_nir"],
        before_swir2_path=bands["before_swir2"],
    ).process()

    assert output.dtype == bool


# --- Pre-existing defects in the same functions --------------------------------


def test_median_accepts_large_kernels_on_16bit_data(tmp_path):
    """
    OpenCV's median filter supports kernels above 5 on uint8 only, so a 7x7
    median -- routine speckle suppression -- raised cv2.error on every 16-bit
    Landsat or Sentinel band, with or without nodata.
    """
    # An independent reference, not the scikit-image path the fix itself uses.
    reference = pytest.importorskip("scipy.ndimage").median_filter

    band = _band(50, 200, 6000, size=20)
    output = fezrs.MedianCalculator(
        tif_path=_write(tmp_path / "pan.tif", band, nodata=None), kernel_size=7
    ).process()

    np.testing.assert_allclose(output, reference(band, size=7, mode="nearest"))


def test_sobel_keeps_the_sign_and_range_of_the_gradient(tmp_path):
    """
    The output depth was 0, which OpenCV reads as CV_8U rather than "same as
    input": on 16-bit data negative gradients were clipped to 0 and the rest
    saturated at 255, erasing the sign lineament mapping depends on.
    """
    band = _band(51, 200, 6000, size=20)
    gradient = fezrs.SobelCalculator(
        tif_path=_write(tmp_path / "pan.tif", band, nodata=None), kernel_size=3
    ).process()

    assert gradient.min() < 0
    assert gradient.max() > 255


def test_geoeye_process_returns_its_result(nodata_bands):
    """process() assigned _output but returned None."""
    calculator = fezrs.GeoeyeCalculator(tif_path=nodata_bands["multiband"], level=0)

    assert calculator.process() is calculator._output
