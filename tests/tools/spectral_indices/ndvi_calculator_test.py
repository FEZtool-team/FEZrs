import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from matplotlib.pyplot import cm

from fezrs.tools.spectral_indices.ndvi_calculator import NDVICalculator


@pytest.fixture
def mock_ndvi_calculator():
    fake_normalized_bands = {
        "nir": np.random.rand(100, 100),
        "red": np.random.rand(100, 100),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands

    with patch(
        "fezrs.tools.spectral_indices.ndvi_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = NDVICalculator(
            nir_path="dummy_nir.tif",
            red_path="dummy_red.tif",
        )

    return calculator


def test_initialization(mock_ndvi_calculator):
    assert mock_ndvi_calculator.normalized_bands is not None
    assert "nir" in mock_ndvi_calculator.normalized_bands
    assert "red" in mock_ndvi_calculator.normalized_bands
    assert mock_ndvi_calculator._output is None


def test_validate_method_exists(mock_ndvi_calculator):
    mock_ndvi_calculator._validate()


def test_process_calculates_ndvi_correctly(mock_ndvi_calculator):
    mock_ndvi_calculator.normalized_bands = {
        "nir": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "red": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    nir = mock_ndvi_calculator.normalized_bands["nir"]
    red = mock_ndvi_calculator.normalized_bands["red"]
    expected = (nir - red) / (nir + red)

    result = mock_ndvi_calculator.process()

    assert np.array_equal(result, expected)
    assert np.array_equal(mock_ndvi_calculator._output, expected)


def test_process_handles_division_by_zero(mock_ndvi_calculator):
    mock_ndvi_calculator.normalized_bands = {
        "nir": np.zeros((100, 100)),
        "red": np.zeros((100, 100)),
    }

    result = mock_ndvi_calculator.process()

    assert result is not None


def test_process_returns_correct_shape(mock_ndvi_calculator):
    mock_ndvi_calculator.normalized_bands = {
        "nir": np.random.rand(150, 200),
        "red": np.random.rand(150, 200),
    }

    result = mock_ndvi_calculator.process()

    assert result.shape == (150, 200)


def test_ndvi_values_range(mock_ndvi_calculator):
    mock_ndvi_calculator.normalized_bands = {
        "nir": np.array([[1.0, 0.0], [0.5, 0.3]]),
        "red": np.array([[0.0, 1.0], [0.5, 0.9]]),
    }

    result = mock_ndvi_calculator.process()

    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_execute_calls_base_execute(mock_ndvi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.ndvi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_ndvi_calculator.execute(
            output_path="output/test.png",
            title="NDVI Test",
            figsize=(20, 15),
            show_axis=True,
            colormap=cm.viridis,
            show_colorbar=False,
            filename_prefix="NDVITool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "NDVI Test",
        (20, 15),
        True,
        cm.viridis,
        False,
        "NDVITool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_ndvi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.ndvi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_ndvi_calculator.execute("output.png")

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
