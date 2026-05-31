import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from fezrs.tools.image_enhancement.equalize_rgb_calculator import (
    EqualizeRGBCalculator,
)


@pytest.fixture
def mock_equalize_rgb():
    calculator = EqualizeRGBCalculator.__new__(EqualizeRGBCalculator)

    calculator.normalized_bands = {
        "red": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "green": np.array([[0.2, 0.3], [0.4, 0.5]]),
        "blue": np.array([[0.3, 0.4], [0.5, 0.6]]),
    }

    calculator._output = None

    yield calculator


def test_process_returns_equalized_rgb_stack(mock_equalize_rgb):
    red_eq = np.ones((2, 2)) * 1
    green_eq = np.ones((2, 2)) * 2
    blue_eq = np.ones((2, 2)) * 3

    with patch(
        "fezrs.tools.image_enhancement.equalize_rgb_calculator.exposure.equalize_hist",
        side_effect=[red_eq, green_eq, blue_eq],
    ) as mock_equalize_hist:
        result = mock_equalize_rgb.process()

    expected = np.stack([red_eq, green_eq, blue_eq], axis=2)

    assert mock_equalize_hist.call_count == 3
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(mock_equalize_rgb._output, expected)


def test_histogram_export_calls_required_methods(mock_equalize_rgb):
    output = np.random.rand(2, 2, 3)

    mock_equalize_rgb._validate = MagicMock()
    mock_equalize_rgb.process = MagicMock()
    mock_equalize_rgb._output = output

    mock_equalize_rgb._add_watermark = MagicMock()
    mock_equalize_rgb._save_histogram_figure = MagicMock()

    result = mock_equalize_rgb.histogram_export(
        output_path="output",
        title="Equalize RGB IE",
    )

    mock_equalize_rgb._validate.assert_called_once()
    mock_equalize_rgb.process.assert_called_once()

    mock_equalize_rgb._add_watermark.assert_called_once()
    mock_equalize_rgb._save_histogram_figure.assert_called_once()

    assert result is mock_equalize_rgb


def test_execute_calls_base_execute(mock_equalize_rgb):
    with patch(
        "fezrs.tools.image_enhancement.equalize_rgb_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_equalize_rgb.execute(
            output_path="output",
            title="Equalize RGB Output",
        )

    mock_execute.assert_called_once()
    assert result == "executed"


# NOTE - These block code for integration test the EqualizeRGBCalculator
# if __name__ == "__main__":
#     red_path = Path.cwd() / "data/Red.tif"
#     green_path = Path.cwd() / "data/Green.tif"
#     blue_path = Path.cwd() / "data/Blue.tif"

#     calculator = EqualizeRGBCalculator(
#         red_path=red_path, green_path=green_path, blue_path=blue_path
#     ).execute("./", title="Equalize-RGB IE")
