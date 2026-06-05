import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fezrs.tools.spectral_indices.afri_calculator import AFRICalculator


@pytest.fixture
def mock_afri_calculator():
    fake_normalized_bands = {
        "nir": np.random.rand(100, 100),
        "swir1": np.random.rand(100, 100),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands

    with patch(
        "fezrs.tools.spectral_indices.afri_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = AFRICalculator(
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
        )

    return calculator


def test_initialization(mock_afri_calculator):
    assert mock_afri_calculator.normalized_bands is not None
    assert "nir" in mock_afri_calculator.normalized_bands
    assert "swir1" in mock_afri_calculator.normalized_bands
    assert mock_afri_calculator._output is None


def test_validate_method_exists(mock_afri_calculator):
    mock_afri_calculator._validate()


def test_process_calculates_afri_correctly(mock_afri_calculator):
    mock_afri_calculator.normalized_bands = {
        "nir": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "swir1": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    nir = mock_afri_calculator.normalized_bands["nir"]
    swir1 = mock_afri_calculator.normalized_bands["swir1"]
    expected = (nir - 0.66) * (swir1 / (nir + (0.66 * swir1)))

    result = mock_afri_calculator.process()

    assert np.array_equal(result, expected)
    assert np.array_equal(mock_afri_calculator._output, expected)


def test_process_handles_zero_division(mock_afri_calculator):
    mock_afri_calculator.normalized_bands = {
        "nir": np.zeros((100, 100)),
        "swir1": np.zeros((100, 100)),
    }

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = mock_afri_calculator.process()

    assert result is not None
    assert not np.any(np.isinf(result))
    assert not any(
        issubclass(record.category, RuntimeWarning) for record in captured_warnings
    )


def test_process_returns_correct_shape(mock_afri_calculator):
    mock_afri_calculator.normalized_bands = {
        "nir": np.random.rand(150, 200),
        "swir1": np.random.rand(150, 200),
    }

    result = mock_afri_calculator.process()

    assert result.shape == (150, 200)


def test_execute_calls_base_execute(mock_afri_calculator):
    with patch(
        "fezrs.tools.spectral_indices.afri_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_afri_calculator.execute(
            output_path="output/test.png",
            title="AFRI Test",
            figsize=(20, 15),
            show_axis=True,
            colormap="viridis",
            show_colorbar=False,
            filename_prefix="AFRITool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "AFRI Test",
        (20, 15),
        True,
        "viridis",
        False,
        "AFRITool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_afri_calculator):
    with patch(
        "fezrs.tools.spectral_indices.afri_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_afri_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (15, 10),
        False,
        "gray",
        True,
        "Tool_output",
        1000,
        "tight",
        True,
    )
    assert result == "executed"
