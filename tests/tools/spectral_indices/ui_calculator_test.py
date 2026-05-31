import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from matplotlib.pyplot import cm

from fezrs.tools.spectral_indices.ui_calculator import UICalculator


@pytest.fixture
def mock_ui_calculator():
    fake_normalized_bands = {
        "nir": np.random.rand(100, 100),
        "swir2": np.random.rand(100, 100),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands

    with patch(
        "fezrs.tools.spectral_indices.ui_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = UICalculator(
            nir_path="dummy_nir.tif",
            swir2_path="dummy_swir2.tif",
        )

    return calculator


def test_initialization(mock_ui_calculator):
    assert mock_ui_calculator.normalized_bands is not None
    assert "nir" in mock_ui_calculator.normalized_bands
    assert "swir2" in mock_ui_calculator.normalized_bands
    assert mock_ui_calculator._output is None


def test_validate_method_exists(mock_ui_calculator):
    mock_ui_calculator._validate()


def test_process_calculates_ui_correctly(mock_ui_calculator):
    mock_ui_calculator.normalized_bands = {
        "nir": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "swir2": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    nir = mock_ui_calculator.normalized_bands["nir"]
    swir2 = mock_ui_calculator.normalized_bands["swir2"]
    expected = (swir2 - nir) / (nir + swir2)

    result = mock_ui_calculator.process()

    assert np.array_equal(result, expected)
    assert np.array_equal(mock_ui_calculator._output, expected)


def test_process_handles_division_by_zero(mock_ui_calculator):
    mock_ui_calculator.normalized_bands = {
        "nir": np.zeros((100, 100)),
        "swir2": np.zeros((100, 100)),
    }

    result = mock_ui_calculator.process()

    assert result is not None


def test_process_returns_correct_shape(mock_ui_calculator):
    mock_ui_calculator.normalized_bands = {
        "nir": np.random.rand(150, 200),
        "swir2": np.random.rand(150, 200),
    }

    result = mock_ui_calculator.process()

    assert result.shape == (150, 200)


def test_ui_values_range(mock_ui_calculator):
    mock_ui_calculator.normalized_bands = {
        "nir": np.array([[1.0, 0.0], [0.5, 0.3]]),
        "swir2": np.array([[0.0, 1.0], [0.5, 0.9]]),
    }

    result = mock_ui_calculator.process()

    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_execute_calls_base_execute(mock_ui_calculator):
    with patch(
        "fezrs.tools.spectral_indices.ui_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_ui_calculator.execute(
            output_path="output/test.png",
            title="UI Test",
            figsize=(20, 15),
            show_axis=True,
            colormap=cm.viridis,
            show_colorbar=False,
            filename_prefix="UITool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "UI Test",
        (20, 15),
        True,
        cm.viridis,
        False,
        "UITool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_ui_calculator):
    with patch(
        "fezrs.tools.spectral_indices.ui_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_ui_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (15, 10),
        False,
        cm.gray,
        True,
        "Tool_output",
        1000,
        "tight",
        True,
    )
    assert result == "executed"
