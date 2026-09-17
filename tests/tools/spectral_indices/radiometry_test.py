"""
Radiometric behaviour of the spectral indices (issue #37).

The indices previously read from ``get_normalized_bands()``, which min-max
rescales each band independently. Because every band received a *different*
affine transform, the relationships between bands were altered -- and those
relationships are the entire physical content of a band ratio. Two consequences
are tested here: a pixel's value depended on how much of the scene was loaded,
and inter-band correlations inverted.
"""

from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.spectral_indices.afri_calculator import AFRICalculator
from fezrs.tools.spectral_indices.ndvi_calculator import NDVICalculator
from fezrs.tools.spectral_indices.savi_calculator import SAVICalculator
from fezrs.utils.radiometry_handler import (
    RADIOMETRIC_PRESETS,
    apply_scaling,
    warn_if_not_reflectance,
)


def _build(calculator_class, module_path, bands, **kwargs):
    handler = MagicMock()
    handler.get_bands.return_value = bands

    def fake_init(self, *args, **inner):
        self.files_handler = handler
        self._output = None

    with patch(f"{module_path}.BaseTool.__init__", fake_init):
        return calculator_class(**kwargs)


def _ndvi(bands, **kwargs):
    return _build(
        NDVICalculator,
        "fezrs.tools.spectral_indices.ndvi_calculator",
        bands,
        nir_path="nir.tif",
        red_path="red.tif",
        **kwargs,
    )


# --- Scene independence -------------------------------------------------------


def test_index_value_does_not_depend_on_loaded_extent():
    """
    The sharpest reproducibility failure in #37: under per-band min-max scaling
    the same pixel produced a different NDVI depending on whether the full scene
    or a crop was loaded, because the rescale used the extent's own extrema.
    """
    rng = np.random.default_rng(7)
    nir = rng.integers(500, 6000, size=(64, 64)).astype(float)
    red = rng.integers(200, 3000, size=(64, 64)).astype(float)

    full = _ndvi({"nir": nir, "red": red}).process()
    crop = _ndvi({"nir": nir[:16, :16], "red": red[:16, :16]}).process()

    np.testing.assert_allclose(full[:16, :16], crop)


def test_normalized_inputs_would_not_be_extent_independent():
    """
    Guards the reasoning above: the same comparison fails under the min-max
    rescale the indices used to apply, so the test is not vacuous.
    """
    rng = np.random.default_rng(7)
    nir = rng.integers(500, 6000, size=(64, 64)).astype(float)
    red = rng.integers(200, 3000, size=(64, 64)).astype(float)

    def normalize(a):
        return (a - a.min()) / (a.max() - a.min())

    def ndvi(n, r):
        return (n - r) / (n + r)

    full = ndvi(normalize(nir), normalize(red))
    crop = ndvi(normalize(nir[:16, :16]), normalize(red[:16, :16]))

    assert not np.allclose(full[:16, :16], crop)


def test_normalized_difference_is_invariant_to_a_common_gain():
    """
    A normalized difference is insensitive to a gain applied to every band
    equally -- which is why NDVI on raw DN is still meaningful, unlike SAVI.
    """
    nir = np.array([[0.5, 0.6]])
    red = np.array([[0.1, 0.2]])

    plain = _ndvi({"nir": nir, "red": red}).process()
    gained = _ndvi({"nir": nir * 10000, "red": red * 10000}).process()

    np.testing.assert_allclose(plain, gained)


EXAMPLE_DATA = Path(__file__).resolve().parents[3] / "example" / "data"


def _read_example_band(name):
    rasterio = pytest.importorskip("rasterio")
    path = EXAMPLE_DATA / f"{name}.tif"
    if not path.is_file():
        pytest.skip(f"bundled example band {name} not available")
    with rasterio.open(path) as source:
        return source.read(1).astype(float)


def _correlate(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def test_afri_tracks_ndvi_on_real_bands():
    """
    Karnieli et al. (2001) report that under clear-sky conditions AFRI and NDVI
    values are "almost identical". That gives an independent physical check on
    the pipeline, and it needs genuinely correlated bands -- synthetic noise has
    no spectral structure to preserve or destroy.

    On the bundled Landsat subset the relationship holds on values as read and
    inverts under the per-band min-max rescale the indices used to apply, which
    is the measurement behind issue #37.
    """
    nir = _read_example_band("nir")
    red = _read_example_band("red")
    swir1 = _read_example_band("swir_1")

    def normalize(a):
        return (a - a.min()) / (a.max() - a.min())

    ndvi = (nir - red) / (nir + red)
    afri = (nir - 0.66 * swir1) / (nir + 0.66 * swir1)

    afri_normalized = (normalize(nir) - 0.66 * normalize(swir1)) / (
        normalize(nir) + 0.66 * normalize(swir1)
    )
    ndvi_normalized = (normalize(nir) - normalize(red)) / (
        normalize(nir) + normalize(red)
    )

    # As read: the physically expected agreement.
    assert _correlate(afri, ndvi) > 0.5
    # Per-band rescaled: the sign flips, so the two disagree about the scene.
    assert _correlate(afri_normalized, ndvi_normalized) < 0


def test_bsi_opposes_vegetation_on_real_bands():
    """
    Bare ground and canopy are inverse surface types, so BSI must correlate
    negatively with NDVI. Under per-band rescaling that correlation turns
    positive, i.e. the index reports bare ground where vegetation is.
    """
    nir = _read_example_band("nir")
    red = _read_example_band("red")
    blue = _read_example_band("blue")
    swir1 = _read_example_band("swir_1")

    def normalize(a):
        return (a - a.min()) / (a.max() - a.min())

    def bsi(s, r, n, b):
        return ((s + r) - (n + b)) / ((s + r) + (n + b))

    ndvi = (nir - red) / (nir + red)

    assert _correlate(bsi(swir1, red, nir, blue), ndvi) < 0
    assert (
        _correlate(
            bsi(normalize(swir1), normalize(red), normalize(nir), normalize(blue)),
            (normalize(nir) - normalize(red)) / (normalize(nir) + normalize(red)),
        )
        > 0
    )


# --- Radiometric scaling ------------------------------------------------------


def test_scaling_converts_landsat_c2l2_integers_to_reflectance():
    digital_numbers = np.array([[10000.0, 20000.0]])

    scaled = apply_scaling(
        {"nir": digital_numbers}, **RADIOMETRIC_PRESETS["landsat-c2-l2"]
    )

    expected = digital_numbers * 2.75e-5 - 0.2
    np.testing.assert_allclose(scaled["nir"], expected)
    assert 0.0 <= scaled["nir"].min() <= 1.0


def test_identity_scaling_leaves_values_untouched():
    bands = {"nir": np.array([[1.0, 2.0]])}

    scaled = apply_scaling(bands, scale_factor=1.0, offset=0.0)

    np.testing.assert_allclose(scaled["nir"], bands["nir"])


def test_calculator_applies_scaling_before_computing():
    nir = np.array([[20000.0]])
    red = np.array([[10000.0]])

    calculator = _ndvi(
        {"nir": nir, "red": red}, **RADIOMETRIC_PRESETS["landsat-c2-l2"]
    )
    result = calculator.process()

    nir_reflectance = nir * 2.75e-5 - 0.2
    red_reflectance = red * 2.75e-5 - 0.2
    expected = (nir_reflectance - red_reflectance) / (
        nir_reflectance + red_reflectance
    )

    np.testing.assert_allclose(result, expected)


def test_savi_warns_when_given_unscaled_digital_numbers():
    """
    SAVI's L = 0.5 is defined in reflectance units. Added to a DN in the
    thousands it contributes nothing, so the result is silently not SAVI.
    """
    bands = {
        "nir": np.array([[4200.0, 5100.0]]),
        "red": np.array([[1200.0, 1500.0]]),
    }

    with pytest.warns(UserWarning, match="outside the reflectance range"):
        _build(
            SAVICalculator,
            "fezrs.tools.spectral_indices.savi_calculator",
            bands,
            nir_path="nir.tif",
            red_path="red.tif",
        )


def test_afri_warns_when_given_unscaled_digital_numbers():
    bands = {
        "nir": np.array([[4200.0]]),
        "swir1": np.array([[2600.0]]),
    }

    with pytest.warns(UserWarning, match="outside the reflectance range"):
        _build(
            AFRICalculator,
            "fezrs.tools.spectral_indices.afri_calculator",
            bands,
            nir_path="nir.tif",
            swir1_path="swir1.tif",
        )


def test_no_warning_once_scaling_is_supplied():
    bands = {
        "nir": np.array([[20000.0]]),
        "red": np.array([[10000.0]]),
    }

    with _no_warnings():
        _build(
            SAVICalculator,
            "fezrs.tools.spectral_indices.savi_calculator",
            bands,
            nir_path="nir.tif",
            red_path="red.tif",
            **RADIOMETRIC_PRESETS["landsat-c2-l2"],
        )


def test_gain_invariant_indices_do_not_warn():
    """
    NDVI carries no reflectance-scale constant, so unscaled input is legitimate
    and must not raise a warning.
    """
    bands = {
        "nir": np.array([[4200.0]]),
        "red": np.array([[1200.0]]),
    }

    with _no_warnings():
        _ndvi(bands)


def test_warning_helper_ignores_gain_invariant_indices():
    bands = {"nir": np.array([[9999.0]])}

    with _no_warnings():
        warn_if_not_reflectance(bands, "NDVI")


class _no_warnings:
    """Assert that no UserWarning is emitted inside the block."""

    def __enter__(self):
        import warnings

        self._manager = warnings.catch_warnings(record=True)
        self._records = self._manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc_info):
        offending = [
            record
            for record in self._records
            if issubclass(record.category, UserWarning)
            and "reflectance range" in str(record.message)
        ]
        self._manager.__exit__(*exc_info)
        assert not offending, f"unexpected warnings: {offending}"
        return False
