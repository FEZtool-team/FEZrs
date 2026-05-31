import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.image_enhancement.float_calculator import FloatCalculator


@pytest.fixture
def mock_float_calculator():
    fake_metadata = {"nir": {"image_skimage": np.random.rand(10, 10)}}

    fake_files_handler = MagicMock()
    fake_files_handler.get_metadata_bands.return_value = fake_metadata

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.metadata_bands = fake_metadata

    with patch(
        "fezrs.tools.image_enhancement.float_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = FloatCalculator(
            nir_path="dummy_nir.tif",
        )

    return calculator


def test_initialization(mock_float_calculator):
    assert mock_float_calculator.metadata_bands is not None
    assert "nir" in mock_float_calculator.metadata_bands
    assert mock_float_calculator._output is None


def test_validate_method_exists(mock_float_calculator):
    mock_float_calculator._validate()


def test_process_returns_float_image(mock_float_calculator):
    expected = np.ones((10, 10), dtype=np.float64)

    with patch(
        "fezrs.tools.image_enhancement.float_calculator.img_as_float",
        return_value=expected,
    ) as mock_img_as_float:
        result = mock_float_calculator.process()

    mock_img_as_float.assert_called_once_with(
        mock_float_calculator.metadata_bands["nir"]["image_skimage"]
    )
    assert np.array_equal(result, expected)
    assert mock_float_calculator._output is not None


def test_histogram_export_calls_required_methods(mock_float_calculator):
    with (
        patch.object(mock_float_calculator, "_validate") as mock_validate,
        patch.object(mock_float_calculator, "process") as mock_process,
        patch.object(mock_float_calculator, "_add_watermark") as mock_watermark,
        patch.object(mock_float_calculator, "_save_histogram_figure") as mock_save,
        patch(
            "fezrs.tools.image_enhancement.float_calculator.plt.subplots"
        ) as mock_subplots,
    ):
        mock_float_calculator._output = np.ones((10, 10))

        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)

        result = mock_float_calculator.histogram_export(
            output_path="output/histogram.png",
            title="Test Float Histogram",
            figsize=(12, 8),
            filename_prefix="Float_Histogram",
            dpi=300,
            bbox_inches="tight",
        )

        mock_validate.assert_called_once()
        mock_process.assert_called_once()
        mock_watermark.assert_called_once_with(ax)
        mock_save.assert_called_once_with(
            ax, "output/histogram.png", "Float_Histogram", 300, "tight"
        )
        assert result is mock_float_calculator


def test_histogram_export_with_default_parameters(mock_float_calculator):
    with (
        patch.object(mock_float_calculator, "_validate") as mock_validate,
        patch.object(mock_float_calculator, "process") as mock_process,
        patch.object(mock_float_calculator, "_add_watermark") as mock_watermark,
        patch.object(mock_float_calculator, "_save_histogram_figure") as mock_save,
        patch(
            "fezrs.tools.image_enhancement.float_calculator.plt.subplots"
        ) as mock_subplots,
    ):
        mock_float_calculator._output = np.random.rand(10, 10)

        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)

        result = mock_float_calculator.histogram_export(
            output_path="output/histogram.png"
        )

        mock_validate.assert_called_once()
        mock_process.assert_called_once()
        mock_watermark.assert_called_once_with(ax)
        mock_save.assert_called_once()
        assert result is mock_float_calculator


def test_histogram_bins_and_density(mock_float_calculator):
    with (
        patch.object(mock_float_calculator, "_validate"),
        patch.object(mock_float_calculator, "process"),
        patch.object(mock_float_calculator, "_add_watermark"),
        patch.object(mock_float_calculator, "_save_histogram_figure"),
        patch(
            "fezrs.tools.image_enhancement.float_calculator.plt.subplots"
        ) as mock_subplots,
    ):
        mock_float_calculator._output = np.random.rand(10, 10)

        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)

        mock_float_calculator.histogram_export(output_path="output.png", title="Test")

        call_args = ax.hist.call_args
        assert call_args[1]["bins"] == 256
        assert call_args[1]["density"]
        assert call_args[1]["histtype"] == "bar"
        assert call_args[1]["color"] == "black"

        ax.ticklabel_format.assert_called_once_with(style="plain")
        ax.set_title.assert_called_once_with("Test-FEZrs")


def test_execute_calls_base_execute(mock_float_calculator):
    with patch(
        "fezrs.tools.image_enhancement.float_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_float_calculator.execute(
            output_path="output/test.png",
            title="Test Image",
            figsize=(15, 10),
            show_axis=True,
            colormap="viridis",
            show_colorbar=True,
            filename_prefix="FloatTool",
            dpi=300,
            bbox_inches="tight",
            grid=False,
            nrows=2,
            ncols=2,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "Test Image",
        (15, 10),
        True,
        "viridis",
        True,
        "FloatTool",
        300,
        "tight",
        False,
        2,
        2,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_float_calculator):
    with patch(
        "fezrs.tools.image_enhancement.float_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_float_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (10, 10),
        False,
        "gray",
        False,
        "Tool_output",
        500,
        "tight",
        True,
        None,
        None,
    )
    assert result == "executed"


def test_process_stores_output_in_instance_variable(mock_float_calculator):
    expected = np.random.rand(10, 10)

    with patch(
        "fezrs.tools.image_enhancement.float_calculator.img_as_float",
        return_value=expected,
    ):
        result = mock_float_calculator.process()

    assert mock_float_calculator._output is not None
    assert np.array_equal(mock_float_calculator._output, expected)
    assert np.array_equal(result, expected)


# NOTE - These block code for integration test the FloatCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = FloatCalculator(nir_path=nir_path).execute("./", title="Float IE")
#     calculator = FloatCalculator(nir_path=nir_path).histogram_export("./")
