import numpy as np
import pytest
import rasterio

from fezrs.tools.spectral_indices.afri_calculator import AFRICalculator
from fezrs.tools.spectral_indices.bi_calculator import BICalculator
from fezrs.tools.spectral_indices.ndvi_calculator import NDVICalculator
from fezrs.tools.spectral_indices.ndwi_calculator import NDWICalculator
from fezrs.tools.spectral_indices.savi_calculator import SAVICalculator
from fezrs.tools.spectral_indices.ui_calculator import UICalculator
from fezrs.tools.spectral_indices._division import divide_with_nan


def _write_tif(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def band_paths(tmp_path):
    # Values outside [0, 1] so a min-max rescale would change the result.
    bands = {
        "nir": np.array([[3000.0, 4000.0], [5000.0, 2000.0]], dtype=np.float32),
        "red": np.array([[1000.0, 1500.0], [2500.0, 3500.0]], dtype=np.float32),
        "green": np.array([[800.0, 1200.0], [1800.0, 2200.0]], dtype=np.float32),
        "swir1": np.array([[1500.0, 1800.0], [2100.0, 900.0]], dtype=np.float32),
        "swir2": np.array([[1100.0, 1600.0], [1900.0, 700.0]], dtype=np.float32),
    }
    paths = {}
    for name, data in bands.items():
        path = tmp_path / f"{name}.tif"
        _write_tif(path, data)
        paths[name] = path
    paths["arrays"] = bands
    return paths


def test_ndvi_matches_standard_formula(band_paths):
    nir = band_paths["arrays"]["nir"]
    red = band_paths["arrays"]["red"]
    expected = divide_with_nan(nir - red, nir + red)

    result = NDVICalculator(
        nir_path=band_paths["nir"],
        red_path=band_paths["red"],
    ).process()

    np.testing.assert_allclose(result, expected)


def test_ndwi_matches_mcfeeters_formula(band_paths):
    nir = band_paths["arrays"]["nir"]
    green = band_paths["arrays"]["green"]
    expected = divide_with_nan(green - nir, nir + green)

    result = NDWICalculator(
        nir_path=band_paths["nir"],
        green_path=band_paths["green"],
    ).process()

    np.testing.assert_allclose(result, expected)


def test_savi_matches_huete_formula(band_paths):
    nir = band_paths["arrays"]["nir"]
    red = band_paths["arrays"]["red"]
    expected = divide_with_nan(nir - red, nir + red + 0.5) * 1.5

    result = SAVICalculator(
        nir_path=band_paths["nir"],
        red_path=band_paths["red"],
    ).process()

    np.testing.assert_allclose(result, expected)


def test_afri_matches_karnieli_2001(band_paths):
    nir = band_paths["arrays"]["nir"]
    swir1 = band_paths["arrays"]["swir1"]
    expected = divide_with_nan(nir - 0.66 * swir1, nir + 0.66 * swir1)

    result = AFRICalculator(
        nir_path=band_paths["nir"],
        swir1_path=band_paths["swir1"],
    ).process()

    np.testing.assert_allclose(result, expected)


def test_ui_matches_standard_formula(band_paths):
    nir = band_paths["arrays"]["nir"]
    swir2 = band_paths["arrays"]["swir2"]
    expected = divide_with_nan(swir2 - nir, nir + swir2)

    result = UICalculator(
        nir_path=band_paths["nir"],
        swir2_path=band_paths["swir2"],
    ).process()

    np.testing.assert_allclose(result, expected)


def test_bi_matches_implemented_formula(band_paths):
    nir = band_paths["arrays"]["nir"]
    red = band_paths["arrays"]["red"]
    green = band_paths["arrays"]["green"]
    expected = divide_with_nan(nir - green - red, nir + green + red)

    result = BICalculator(
        nir_path=band_paths["nir"],
        red_path=band_paths["red"],
        green_path=band_paths["green"],
    ).process()

    np.testing.assert_allclose(result, expected)


def test_ndvi_is_independent_of_scene_extent(tmp_path):
    nir_full = np.array([[3000.0, 4000.0], [100.0, 200.0]], dtype=np.float32)
    red_full = np.array([[1000.0, 1500.0], [800.0, 900.0]], dtype=np.float32)
    nir_crop = nir_full[:1, :1]
    red_crop = red_full[:1, :1]

    full_nir = tmp_path / "nir_full.tif"
    full_red = tmp_path / "red_full.tif"
    crop_nir = tmp_path / "nir_crop.tif"
    crop_red = tmp_path / "red_crop.tif"
    _write_tif(full_nir, nir_full)
    _write_tif(full_red, red_full)
    _write_tif(crop_nir, nir_crop)
    _write_tif(crop_red, red_crop)

    full = NDVICalculator(nir_path=full_nir, red_path=full_red).process()
    crop = NDVICalculator(nir_path=crop_nir, red_path=crop_red).process()

    np.testing.assert_allclose(full[0, 0], crop[0, 0])
