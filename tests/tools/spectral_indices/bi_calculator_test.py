import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fezrs.tools.spectral_indices.bi_calculator import BICalculator


@pytest.fixture
def mock_bi_calculator():
    fake_normalized_bands = {
        "nir": np.random.rand(100, 100),
        "red": np.random.rand(100, 100),
        "green": np.random.rand(100, 100),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands

    with patch(
        "fezrs.tools.spectral_indices.bi_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = BICalculator(
            nir_path="dummy_nir.tif",
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
        )

    return calculator


def test_initialization(mock_bi_calculator):
    assert mock_bi_calculator.normalized_bands is not None
    assert "nir" in mock_bi_calculator.normalized_bands
    assert "red" in mock_bi_calculator.normalized_bands
    assert "green" in mock_bi_calculator.normalized_bands
    assert mock_bi_calculator._output is None


def test_validate_method_exists(mock_bi_calculator):
    mock_bi_calculator._validate()


def test_process_calculates_bi_correctly(mock_bi_calculator):
    mock_bi_calculator.normalized_bands = {
        "nir": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "red": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "green": np.array([[0.05, 0.15], [0.25, 0.35]]),
    }

    nir = mock_bi_calculator.normalized_bands["nir"]
    red = mock_bi_calculator.normalized_bands["red"]
    green = mock_bi_calculator.normalized_bands["green"]
    expected = ((nir - green) - red) / ((nir + green) + red)

    result = mock_bi_calculator.process()

    assert np.array_equal(result, expected)
    assert np.array_equal(mock_bi_calculator._output, expected)


def test_process_handles_division_by_zero(mock_bi_calculator):
    mock_bi_calculator.normalized_bands = {
        "nir": np.zeros((100, 100)),
        "red": np.zeros((100, 100)),
        "green": np.zeros((100, 100)),
    }

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = mock_bi_calculator.process()

    assert result is not None
    assert not any(
        issubclass(record.category, RuntimeWarning) for record in captured_warnings
    )


def test_process_returns_correct_shape(mock_bi_calculator):
    mock_bi_calculator.normalized_bands = {
        "nir": np.random.rand(150, 200),
        "red": np.random.rand(150, 200),
        "green": np.random.rand(150, 200),
    }

    result = mock_bi_calculator.process()

    assert result.shape == (150, 200)


def test_execute_calls_base_execute(mock_bi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.bi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_bi_calculator.execute(
            output_path="output/test.png",
            title="BI Test",
            figsize=(20, 15),
            show_axis=True,
            colormap="viridis",
            show_colorbar=False,
            filename_prefix="BITool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "BI Test",
        (20, 15),
        True,
        "viridis",
        False,
        "BITool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_bi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.bi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_bi_calculator.execute("output.png")

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
