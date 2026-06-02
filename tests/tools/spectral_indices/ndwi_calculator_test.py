import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib.pyplot import cm

from fezrs.tools.spectral_indices.ndwi_calculator import NDWICalculator


@pytest.fixture
def mock_ndwi_calculator():
    fake_normalized_bands = {
        "nir": np.random.rand(100, 100),
        "green": np.random.rand(100, 100),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands

    with patch(
        "fezrs.tools.spectral_indices.ndwi_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = NDWICalculator(
            nir_path="dummy_nir.tif",
            green_path="dummy_green.tif",
        )

    return calculator


def test_initialization(mock_ndwi_calculator):
    assert mock_ndwi_calculator.normalized_bands is not None
    assert "nir" in mock_ndwi_calculator.normalized_bands
    assert "green" in mock_ndwi_calculator.normalized_bands
    assert mock_ndwi_calculator._output is None


def test_validate_method_exists(mock_ndwi_calculator):
    mock_ndwi_calculator._validate()


def test_process_calculates_ndwi_correctly(mock_ndwi_calculator):
    mock_ndwi_calculator.normalized_bands = {
        "nir": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "green": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    nir = mock_ndwi_calculator.normalized_bands["nir"]
    green = mock_ndwi_calculator.normalized_bands["green"]
    expected = (green - nir) / (nir + green)

    result = mock_ndwi_calculator.process()

    assert np.array_equal(result, expected)
    assert np.array_equal(mock_ndwi_calculator._output, expected)


def test_process_handles_division_by_zero(mock_ndwi_calculator):
    mock_ndwi_calculator.normalized_bands = {
        "nir": np.zeros((100, 100)),
        "green": np.zeros((100, 100)),
    }

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = mock_ndwi_calculator.process()

    assert result is not None
    assert not any(
        issubclass(record.category, RuntimeWarning) for record in captured_warnings
    )


def test_process_returns_correct_shape(mock_ndwi_calculator):
    mock_ndwi_calculator.normalized_bands = {
        "nir": np.random.rand(150, 200),
        "green": np.random.rand(150, 200),
    }

    result = mock_ndwi_calculator.process()

    assert result.shape == (150, 200)


def test_ndwi_values_range(mock_ndwi_calculator):
    mock_ndwi_calculator.normalized_bands = {
        "nir": np.array([[1.0, 0.0], [0.5, 0.3]]),
        "green": np.array([[0.0, 1.0], [0.5, 0.9]]),
    }

    result = mock_ndwi_calculator.process()

    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_execute_calls_base_execute(mock_ndwi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.ndwi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_ndwi_calculator.execute(
            output_path="output/test.png",
            title="NDWI Test",
            figsize=(20, 15),
            show_axis=True,
            colormap=cm.viridis,
            show_colorbar=False,
            filename_prefix="NDWITool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "NDWI Test",
        (20, 15),
        True,
        cm.viridis,
        False,
        "NDWITool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_ndwi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.ndwi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_ndwi_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (15, 10),
        False,
        cm.Grays,
        True,
        "Tool_output",
        1000,
        "tight",
        True,
    )
    assert result == "executed"
