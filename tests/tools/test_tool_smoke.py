import numpy as np
import rasterio

from fezrs import (
    MagDirCalculator,
    NDVICalculator,
    PCACalculator,
    SpectralProfileCalculator,
)


def _write_tif(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def _six_band_paths(tmp_path):
    paths = {}
    for index, name in enumerate(
        ("red", "green", "blue", "nir", "swir1", "swir2"), start=1
    ):
        data = np.arange(12, dtype=np.float32).reshape(3, 4) * index + 10
        path = tmp_path / f"{name}.tif"
        _write_tif(path, data)
        paths[name] = path
    return paths


def test_ndvi_execute_and_export_raster(tmp_path):
    paths = _six_band_paths(tmp_path)
    calculator = NDVICalculator(nir_path=paths["nir"], red_path=paths["red"])
    calculator.execute(output_path=tmp_path / "figures")
    raster = calculator.export_raster(tmp_path / "rasters", filename="ndvi.tif")

    assert list((tmp_path / "figures").glob("*.png"))
    with rasterio.open(raster) as src:
        assert src.count == 1
        assert src.crs.to_epsg() == 4326
        assert src.read(1).shape == (3, 4)


def test_spectral_profile_histogram_export_writes_png(tmp_path):
    paths = _six_band_paths(tmp_path)
    calculator = SpectralProfileCalculator(
        red_path=paths["red"],
        green_path=paths["green"],
        blue_path=paths["blue"],
        nir_path=paths["nir"],
        swir1_path=paths["swir1"],
        swir2_path=paths["swir2"],
    )
    calculator.histogram_export(output_path=tmp_path / "profile")
    assert list((tmp_path / "profile").glob("*.png"))
    assert calculator.xaxis
    assert calculator.yaxis


def test_pca_histogram_export_without_prior_execute(tmp_path):
    paths = _six_band_paths(tmp_path)
    calculator = PCACalculator(
        red_path=paths["red"],
        green_path=paths["green"],
        blue_path=paths["blue"],
        nir_path=paths["nir"],
        swir1_path=paths["swir1"],
        swir2_path=paths["swir2"],
        component=1,
    )
    calculator.histogram_export(output_path=tmp_path / "pca")
    assert list((tmp_path / "pca").glob("*.png"))
    assert calculator._output is not None
    np.testing.assert_allclose(calculator.explained_variance_ratio.sum(), 1.0)


def test_magdir_no_change_scene(tmp_path):
    paths = _six_band_paths(tmp_path)
    calculator = MagDirCalculator(
        nir_path=paths["nir"],
        swir1_path=paths["swir1"],
        before_nir_path=paths["nir"],
        before_swir1_path=paths["swir1"],
        selecte="magnitude",
    )
    calculator.execute(output_path=tmp_path / "magdir")
    np.testing.assert_allclose(calculator._output, 0)
