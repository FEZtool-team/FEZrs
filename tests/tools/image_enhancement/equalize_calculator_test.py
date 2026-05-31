import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from fezrs.tools.image_enhancement.equalize_calculator import (
    EqualizeCalculator,
)


@pytest.fixture
def mock_equalize():
    with patch(
        "fezrs.tools.image_enhancement.equalize_calculator.BaseTool.__init__",
        return_value=None,
    ):
        calculator = EqualizeCalculator.__new__(EqualizeCalculator)

        calculator.metadata_bands = {
            "nir": {
                "image_skimage": np.ones((10, 10)),
            }
        }

        calculator._output = None

        yield calculator


def test_process_returns_equalized_image(mock_equalize):
    with patch(
        "fezrs.tools.image_enhancement.equalize_calculator.exposure.equalize_hist",
        return_value=np.ones((10, 10)),
    ) as mock_equalize_hist:
        result = mock_equalize.process()

        mock_equalize_hist.assert_called_once()

        assert result.shape == (10, 10)
        assert np.array_equal(result, np.ones((10, 10)))


def test_histogram_export_calls_required_methods(mock_equalize):
    output = np.ones((10, 10))

    mock_equalize.process = MagicMock()
    mock_equalize._output = output

    mock_equalize._validate = MagicMock()
    mock_equalize._add_watermark = MagicMock()
    mock_equalize._save_histogram_figure = MagicMock()

    result = mock_equalize.histogram_export(
        output_path="output",
        title="Equalized",
    )

    mock_equalize._validate.assert_called_once()
    mock_equalize.process.assert_called_once()
    mock_equalize._add_watermark.assert_called_once()
    mock_equalize._save_histogram_figure.assert_called_once()

    assert result is mock_equalize


def test_execute_calls_base_execute(mock_equalize):
    with patch(
        "fezrs.tools.image_enhancement.equalize_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_equalize.execute("output")

        mock_execute.assert_called_once()
        assert result == "executed"


# NOTE - These block code for integration test the EqualizeCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = EqualizeCalculator(nir_path=nir_path).histogram_export(
#         "./", title="Equalize IE"
#     )
