import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.image_enhancement.adaptive_calculator import AdaptiveCalculator


@pytest.fixture
def mock_adaptive():
    fake_metadata = {"nir": {"image_skimage": np.ones((10, 10))}}

    fake_files_handler = MagicMock()
    fake_files_handler.get_metadata_bands.return_value = fake_metadata

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None

    with patch(
        "fezrs.tools.image_enhancement.adaptive_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = AdaptiveCalculator(
            nir_path="dummy.tif",
            clip_limit=0.08,
        )

    return calculator


def test_process_returns_adaptive_equalized_image(mock_adaptive):
    expected = np.ones((10, 10))

    with patch(
        "fezrs.tools.image_enhancement.adaptive_calculator.exposure.equalize_adapthist",
        return_value=expected,
    ) as mock_equalize:
        result = mock_adaptive.process()

    mock_equalize.assert_called_once()
    assert np.array_equal(result, expected)


def test_histogram_export_calls_required_methods(mock_adaptive):
    with (
        patch.object(mock_adaptive, "_validate") as mock_validate,
        patch.object(mock_adaptive, "process") as mock_process,
        patch.object(mock_adaptive, "_add_watermark") as mock_watermark,
        patch.object(mock_adaptive, "_save_histogram_figure") as mock_save,
        patch(
            "fezrs.tools.image_enhancement.adaptive_calculator.plt.subplots"
        ) as mock_subplots,
    ):
        mock_adaptive._output = np.ones((10, 10))

        fig = MagicMock()
        ax = MagicMock()

        mock_subplots.return_value = (fig, ax)

        result = mock_adaptive.histogram_export(
            output_path="output",
            title="Test",
        )

        mock_validate.assert_called_once()
        mock_process.assert_called_once()
        mock_watermark.assert_called_once_with(ax)
        mock_save.assert_called_once()

        assert result is mock_adaptive


def test_execute_calls_base_execute(mock_adaptive):
    with patch(
        "fezrs.tools.image_enhancement.adaptive_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_adaptive.execute("output")

    mock_execute.assert_called_once()
    assert result == "executed"
