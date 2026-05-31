import numpy as np
import pytest

from unittest.mock import patch, MagicMock

from fezrs.tools.image_enhancement.adaptive_rgb_calculator import (
    AdaptiveRGBCalculator,
)


@pytest.fixture
def mock_adaptive_rgb():
    calculator = AdaptiveRGBCalculator.__new__(AdaptiveRGBCalculator)

    calculator.normalized_bands = {
        "red": np.ones((10, 10)),
        "green": np.ones((10, 10)) * 2,
        "blue": np.ones((10, 10)) * 3,
    }

    calculator._output = None

    return calculator


def test_process_returns_adaptive_rgb_stack(mock_adaptive_rgb):
    expected_red = np.full((10, 10), 10)
    expected_green = np.full((10, 10), 20)
    expected_blue = np.full((10, 10), 30)

    with patch(
        "fezrs.tools.image_enhancement.adaptive_rgb_calculator.exposure.equalize_adapthist",
        side_effect=[
            expected_red,
            expected_green,
            expected_blue,
        ],
    ) as mock_equalize:
        result = mock_adaptive_rgb.process()

    assert result.shape == (10, 10, 3)

    np.testing.assert_array_equal(result[:, :, 0], expected_red)
    np.testing.assert_array_equal(result[:, :, 1], expected_green)
    np.testing.assert_array_equal(result[:, :, 2], expected_blue)

    assert mock_equalize.call_count == 3


def test_histogram_export_calls_required_methods(mock_adaptive_rgb):
    mock_adaptive_rgb._output = np.ones((10, 10, 3))

    mock_adaptive_rgb.process = MagicMock(return_value=mock_adaptive_rgb._output)

    mock_adaptive_rgb._validate = MagicMock()
    mock_adaptive_rgb._add_watermark = MagicMock()
    mock_adaptive_rgb._save_histogram_figure = MagicMock()

    result = mock_adaptive_rgb.histogram_export(
        output_path="output",
        title="Adaptive RGB",
    )

    mock_adaptive_rgb._validate.assert_called_once()
    mock_adaptive_rgb.process.assert_called_once()
    mock_adaptive_rgb._add_watermark.assert_called_once()
    mock_adaptive_rgb._save_histogram_figure.assert_called_once()

    assert result is mock_adaptive_rgb


def test_execute_calls_base_execute(mock_adaptive_rgb):
    with patch(
        "fezrs.tools.image_enhancement.adaptive_rgb_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_adaptive_rgb.execute(output_path="output")

    assert result == "executed"
    mock_execute.assert_called_once()


# NOTE - These block code for integration test the AdaptiveRGBCalculator
# if __name__ == "__main__":
#     red_path = Path.cwd() / "data/Red.tif"
#     green_path = Path.cwd() / "data/Green.tif"
#     blue_path = Path.cwd() / "data/Blue.tif"

#     calculator = AdaptiveRGBCalculator(
#         red_path=red_path, green_path=green_path, blue_path=blue_path
#     ).histogram_export("./", title="Adaptive-RGB IE")
