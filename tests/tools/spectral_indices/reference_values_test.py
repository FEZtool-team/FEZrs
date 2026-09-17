"""
Numerical checks of every spectral index against its reference definition.

The existing per-calculator tests verify shapes and execution but re-derive the
expected value from the same expression the implementation uses, so a wrong
formula passes. These tests hard-code values computed by hand from the cited
literature on a 2x2 input, which is what catches a formula that has drifted from
its reference -- the failure mode behind issue #40.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.spectral_indices.afri_calculator import AFRICalculator
from fezrs.tools.spectral_indices.bi_calculator import BICalculator
from fezrs.tools.spectral_indices.ndvi_calculator import NDVICalculator
from fezrs.tools.spectral_indices.ndwi_calculator import NDWICalculator
from fezrs.tools.spectral_indices.savi_calculator import SAVICalculator
from fezrs.tools.spectral_indices.ui_calculator import UICalculator


# Reflectance-scale inputs, chosen so every hand-computed result is exact.
NIR = np.array([[0.50, 0.60], [0.70, 0.80]])
RED = np.array([[0.10, 0.20], [0.30, 0.40]])
GREEN = np.array([[0.15, 0.25], [0.35, 0.45]])
BLUE = np.array([[0.05, 0.10], [0.15, 0.20]])
SWIR1 = np.array([[0.20, 0.30], [0.40, 0.50]])
SWIR2 = np.array([[0.25, 0.35], [0.45, 0.55]])

BANDS = {
    "nir": NIR,
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
    "swir1": SWIR1,
    "swir2": SWIR2,
}


def _build(calculator_class, module_path, **kwargs):
    """Instantiate a calculator with BaseTool.__init__ patched out."""
    handler = MagicMock()
    handler.get_bands.return_value = BANDS

    def fake_init(self, *args, **inner):
        self.files_handler = handler
        self._output = None

    with patch(f"{module_path}.BaseTool.__init__", fake_init):
        calculator = calculator_class(**kwargs)

    calculator.source_bands = BANDS
    return calculator


def test_ndvi_matches_rouse_1974():
    """NDVI = (NIR - Red) / (NIR + Red)."""
    calculator = _build(
        NDVICalculator,
        "fezrs.tools.spectral_indices.ndvi_calculator",
        nir_path="nir.tif",
        red_path="red.tif",
    )

    expected = np.array(
        [
            [(0.50 - 0.10) / (0.50 + 0.10), (0.60 - 0.20) / (0.60 + 0.20)],
            [(0.70 - 0.30) / (0.70 + 0.30), (0.80 - 0.40) / (0.80 + 0.40)],
        ]
    )

    np.testing.assert_allclose(calculator.process(), expected)


def test_ndwi_matches_mcfeeters_1996():
    """NDWI = (Green - NIR) / (Green + NIR)."""
    calculator = _build(
        NDWICalculator,
        "fezrs.tools.spectral_indices.ndwi_calculator",
        nir_path="nir.tif",
        green_path="green.tif",
    )

    expected = np.array(
        [
            [(0.15 - 0.50) / (0.15 + 0.50), (0.25 - 0.60) / (0.25 + 0.60)],
            [(0.35 - 0.70) / (0.35 + 0.70), (0.45 - 0.80) / (0.45 + 0.80)],
        ]
    )

    np.testing.assert_allclose(calculator.process(), expected)


def test_savi_matches_huete_1988():
    """SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L), with L = 0.5."""
    calculator = _build(
        SAVICalculator,
        "fezrs.tools.spectral_indices.savi_calculator",
        nir_path="nir.tif",
        red_path="red.tif",
    )

    expected = np.array(
        [
            [
                (0.50 - 0.10) / (0.50 + 0.10 + 0.5) * 1.5,
                (0.60 - 0.20) / (0.60 + 0.20 + 0.5) * 1.5,
            ],
            [
                (0.70 - 0.30) / (0.70 + 0.30 + 0.5) * 1.5,
                (0.80 - 0.40) / (0.80 + 0.40 + 0.5) * 1.5,
            ],
        ]
    )

    np.testing.assert_allclose(calculator.process(), expected)


def test_ui_matches_reference():
    """UI = (SWIR2 - NIR) / (SWIR2 + NIR)."""
    calculator = _build(
        UICalculator,
        "fezrs.tools.spectral_indices.ui_calculator",
        nir_path="nir.tif",
        swir2_path="swir2.tif",
    )

    expected = np.array(
        [
            [(0.25 - 0.50) / (0.25 + 0.50), (0.35 - 0.60) / (0.35 + 0.60)],
            [(0.45 - 0.70) / (0.45 + 0.70), (0.55 - 0.80) / (0.55 + 0.80)],
        ]
    )

    np.testing.assert_allclose(calculator.process(), expected)


def test_afri_16_matches_karnieli_2001():
    """AFRI_1.6 = (NIR - 0.66*SWIR1) / (NIR + 0.66*SWIR1)."""
    calculator = _build(
        AFRICalculator,
        "fezrs.tools.spectral_indices.afri_calculator",
        nir_path="nir.tif",
        swir1_path="swir1.tif",
    )

    expected = np.array(
        [
            [
                (0.50 - 0.66 * 0.20) / (0.50 + 0.66 * 0.20),
                (0.60 - 0.66 * 0.30) / (0.60 + 0.66 * 0.30),
            ],
            [
                (0.70 - 0.66 * 0.40) / (0.70 + 0.66 * 0.40),
                (0.80 - 0.66 * 0.50) / (0.80 + 0.66 * 0.50),
            ],
        ]
    )

    np.testing.assert_allclose(calculator.process(), expected)


def test_afri_21_matches_karnieli_2001():
    """AFRI_2.1 = (NIR - 0.50*SWIR2) / (NIR + 0.50*SWIR2)."""
    calculator = _build(
        AFRICalculator,
        "fezrs.tools.spectral_indices.afri_calculator",
        nir_path="nir.tif",
        swir2_path="swir2.tif",
        variant="2.1",
    )

    expected = np.array(
        [
            [
                (0.50 - 0.50 * 0.25) / (0.50 + 0.50 * 0.25),
                (0.60 - 0.50 * 0.35) / (0.60 + 0.50 * 0.35),
            ],
            [
                (0.70 - 0.50 * 0.45) / (0.70 + 0.50 * 0.45),
                (0.80 - 0.50 * 0.55) / (0.80 + 0.50 * 0.55),
            ],
        ]
    )

    np.testing.assert_allclose(calculator.process(), expected)


def test_afri_is_bounded_by_plus_minus_one():
    """
    A normalized difference cannot leave [-1, 1] for non-negative reflectance.
    The previous implementation produced 96% negative values on the bundled
    example while the documentation claimed a [0, 1] range.
    """
    calculator = _build(
        AFRICalculator,
        "fezrs.tools.spectral_indices.afri_calculator",
        nir_path="nir.tif",
        swir1_path="swir1.tif",
    )

    output = calculator.process()

    assert np.nanmin(output) >= -1.0
    assert np.nanmax(output) <= 1.0


def test_afri_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Invalid AFRI variant"):
        _build(
            AFRICalculator,
            "fezrs.tools.spectral_indices.afri_calculator",
            nir_path="nir.tif",
            swir1_path="swir1.tif",
            variant="1.7",
        )


def test_afri_variant_requires_its_band():
    with pytest.raises(ValueError, match="requires swir2_path"):
        _build(
            AFRICalculator,
            "fezrs.tools.spectral_indices.afri_calculator",
            nir_path="nir.tif",
            swir1_path="swir1.tif",
            variant="2.1",
        )


def test_bsi_matches_rikimaru_2002():
    """BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))."""
    calculator = _build(
        BICalculator,
        "fezrs.tools.spectral_indices.bi_calculator",
        nir_path="nir.tif",
        red_path="red.tif",
        swir1_path="swir1.tif",
        blue_path="blue.tif",
    )

    assert calculator.formulation == "bsi"

    numerator = (SWIR1 + RED) - (NIR + BLUE)
    denominator = (SWIR1 + RED) + (NIR + BLUE)

    np.testing.assert_allclose(calculator.process(), numerator / denominator)


def test_bi_legacy_formulation_is_preserved_and_warns():
    """
    The previous expression stays reachable so existing results remain
    reproducible, but it is flagged as not a published bare-soil index.
    """
    with pytest.warns(DeprecationWarning, match="not a published bare-soil index"):
        calculator = _build(
            BICalculator,
            "fezrs.tools.spectral_indices.bi_calculator",
            nir_path="nir.tif",
            red_path="red.tif",
            green_path="green.tif",
        )

    assert calculator.formulation == "legacy"

    expected = ((NIR - GREEN) - RED) / ((NIR + GREEN) + RED)

    np.testing.assert_allclose(calculator.process(), expected)


def test_bsi_separates_bare_ground_from_vegetation():
    """
    Physical sanity check: vegetation is bright in NIR and dark in Red/SWIR1,
    bare rock is the inverse. BSI must be negative over the first and positive
    over the second.
    """
    handler = MagicMock()
    vegetation_then_rock = {
        "swir1": np.array([[0.10, 0.40]]),
        "red": np.array([[0.04, 0.30]]),
        "nir": np.array([[0.55, 0.28]]),
        "blue": np.array([[0.02, 0.20]]),
    }
    handler.get_bands.return_value = vegetation_then_rock

    def fake_init(self, *args, **kwargs):
        self.files_handler = handler
        self._output = None

    with patch(
        "fezrs.tools.spectral_indices.bi_calculator.BaseTool.__init__", fake_init
    ):
        calculator = BICalculator(
            nir_path="nir.tif",
            red_path="red.tif",
            swir1_path="swir1.tif",
            blue_path="blue.tif",
        )
    calculator.source_bands = vegetation_then_rock

    output = calculator.process()

    assert output[0, 0] < 0  # vegetation
    assert output[0, 1] > 0  # bare rock
