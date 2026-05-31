import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.spectral_indices.afvi_calculator import AFVICalculator


@pytest.fixture
def mock_afvi_calculator():
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
        "fezrs.tools.spectral_indices.afvi_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = AFVICalculator(
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
        )

    return calculator


def test_initialization(mock_afvi_calculator):
    assert mock_afvi_calculator.normalized_bands is not None
    assert "nir" in mock_afvi_calculator.normalized_bands
    assert "swir1" in mock_afvi_calculator.normalized_bands
    assert mock_afvi_calculator._output is None


def test_validate_method_exists(mock_afvi_calculator):
    mock_afvi_calculator._validate()


def test_process_calculates_afvi_correctly(mock_afvi_calculator):
    mock_afvi_calculator.normalized_bands = {
        "nir": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "swir1": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }

    nir = mock_afvi_calculator.normalized_bands["nir"]
    swir1 = mock_afvi_calculator.normalized_bands["swir1"]
    expected = (nir - 0.66) * (swir1 / (nir + (0.66 * swir1)))

    result = mock_afvi_calculator.process()

    assert np.array_equal(result, expected)
    assert np.array_equal(mock_afvi_calculator._output, expected)


def test_process_handles_zero_division(mock_afvi_calculator):
    mock_afvi_calculator.normalized_bands = {
        "nir": np.zeros((100, 100)),
        "swir1": np.ones((100, 100)),
    }

    result = mock_afvi_calculator.process()

    assert result is not None
    assert not np.any(np.isinf(result))


def test_process_returns_correct_shape(mock_afvi_calculator):
    mock_afvi_calculator.normalized_bands = {
        "nir": np.random.rand(150, 200),
        "swir1": np.random.rand(150, 200),
    }

    result = mock_afvi_calculator.process()

    assert result.shape == (150, 200)


def test_execute_calls_base_execute(mock_afvi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.afvi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_afvi_calculator.execute(
            output_path="output/test.png",
            title="AFVI Test",
            figsize=(20, 15),
            show_axis=True,
            colormap="viridis",
            show_colorbar=False,
            filename_prefix="AFVITool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "AFVI Test",
        (20, 15),
        True,
        "viridis",
        False,
        "AFVITool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_afvi_calculator):
    with patch(
        "fezrs.tools.spectral_indices.afvi_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_afvi_calculator.execute("output.png")

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
