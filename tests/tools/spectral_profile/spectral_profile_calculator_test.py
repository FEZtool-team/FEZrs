import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.spectral_profile.spectral_profile_calculator import (
    SpectralProfileCalculator,
)


@pytest.fixture
def mock_spectral_profile_calculator():
    fake_bands = {
        "red": np.random.rand(100, 100),
        "green": np.random.rand(100, 100),
        "blue": np.random.rand(100, 100),
        "nir": np.random.rand(100, 100),
        "swir1": np.random.rand(100, 100),
        "swir2": np.random.rand(100, 100),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.bands = fake_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.xaxis = []
        self.yaxis = []

    with patch(
        "fezrs.tools.spectral_profile.spectral_profile_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = SpectralProfileCalculator(
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
            blue_path="dummy_blue.tif",
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
            swir2_path="dummy_swir2.tif",
        )

    return calculator


def test_initialization(mock_spectral_profile_calculator):
    assert mock_spectral_profile_calculator.files_handler is not None
    assert mock_spectral_profile_calculator._output is None
    assert mock_spectral_profile_calculator.xaxis == []
    assert mock_spectral_profile_calculator.yaxis == []


def test_validate_method_exists(mock_spectral_profile_calculator):
    mock_spectral_profile_calculator._validate()


def test_process_applies_log_adjustment(mock_spectral_profile_calculator):
    with patch(
        "fezrs.tools.spectral_profile.spectral_profile_calculator.exposure.adjust_log"
    ) as mock_adjust_log:
        mock_adjust_log.return_value = np.random.rand(100, 100)

        mock_spectral_profile_calculator.process()

        mock_adjust_log.assert_called_once()
        assert mock_spectral_profile_calculator._output is not None


def test_process_raises_error_when_index_4_is_invalid(mock_spectral_profile_calculator):
    mock_spectral_profile_calculator.files_handler.bands = {
        "red": np.random.rand(100, 100),
        "green": np.random.rand(100, 100),
        "blue": np.random.rand(100, 100),
        "nir": np.random.rand(100, 100),
    }

    with pytest.raises(ValueError, match="Invalid image data at index 4."):
        mock_spectral_profile_calculator.process()


def test_process_creates_xaxis_and_yaxis(mock_spectral_profile_calculator):
    with patch(
        "fezrs.tools.spectral_profile.spectral_profile_calculator.exposure.adjust_log"
    ):
        mock_spectral_profile_calculator.process()

        assert len(mock_spectral_profile_calculator.xaxis) == 6
        assert len(mock_spectral_profile_calculator.yaxis) == 6
        assert all(isinstance(x, str) for x in mock_spectral_profile_calculator.xaxis)
        assert all(
            isinstance(y, (float, np.float64))
            for y in mock_spectral_profile_calculator.yaxis
        )


def test_process_calculates_mean_for_each_band(mock_spectral_profile_calculator):
    mock_spectral_profile_calculator.files_handler.bands = {
        "red": np.ones((10, 10)) * 0.5,
        "green": np.ones((10, 10)) * 0.6,
        "blue": np.ones((10, 10)) * 0.7,
        "nir": np.ones((10, 10)) * 0.8,
        "swir1": np.ones((10, 10)) * 0.9,
        "swir2": np.ones((10, 10)) * 1.0,
    }

    with patch(
        "fezrs.tools.spectral_profile.spectral_profile_calculator.exposure.adjust_log"
    ):
        mock_spectral_profile_calculator.process()

        expected_means = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert np.allclose(mock_spectral_profile_calculator.yaxis, expected_means)


def test_histogram_export_calls_required_methods(mock_spectral_profile_calculator):
    with (
        patch.object(mock_spectral_profile_calculator, "_validate") as mock_validate,
        patch.object(mock_spectral_profile_calculator, "process") as mock_process,
        patch.object(
            mock_spectral_profile_calculator, "_add_watermark"
        ) as mock_watermark,
        patch.object(
            mock_spectral_profile_calculator, "_save_histogram_figure"
        ) as mock_save,
        patch(
            "fezrs.tools.spectral_profile.spectral_profile_calculator.plt.subplots"
        ) as mock_subplots,
    ):
        mock_spectral_profile_calculator.xaxis = ["red", "green", "blue"]
        mock_spectral_profile_calculator.yaxis = [0.5, 0.6, 0.7]

        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)

        result = mock_spectral_profile_calculator.histogram_export(
            output_path="./output.png",
            title="Spectral Profile Test",
            grid=True,
        )

        mock_validate.assert_called_once()
        mock_process.assert_called_once()
        ax.plot.assert_called_once_with(
            mock_spectral_profile_calculator.xaxis,
            mock_spectral_profile_calculator.yaxis,
        )
        mock_watermark.assert_called_once_with(ax)
        mock_save.assert_called_once()
        assert result is mock_spectral_profile_calculator


def test_histogram_export_sets_title_correctly(mock_spectral_profile_calculator):
    with (
        patch.object(mock_spectral_profile_calculator, "_validate"),
        patch.object(mock_spectral_profile_calculator, "process"),
        patch.object(mock_spectral_profile_calculator, "_add_watermark"),
        patch.object(mock_spectral_profile_calculator, "_save_histogram_figure"),
        patch(
            "fezrs.tools.spectral_profile.spectral_profile_calculator.plt.subplots"
        ) as mock_subplots,
        patch(
            "fezrs.tools.spectral_profile.spectral_profile_calculator.plt.title"
        ) as mock_plt_title,
    ):
        mock_spectral_profile_calculator.xaxis = ["red", "green", "blue"]
        mock_spectral_profile_calculator.yaxis = [0.5, 0.6, 0.7]

        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)

        mock_spectral_profile_calculator.histogram_export(
            output_path="./output.png",
            title="My Spectral Analysis",
            grid=True,
        )

        mock_plt_title.assert_called_once_with("My Spectral Analysis-FEZrs")


def test_execute_calls_base_execute(mock_spectral_profile_calculator):
    with patch(
        "fezrs.tools.spectral_profile.spectral_profile_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_spectral_profile_calculator.execute(
            output_path="output/test.png",
            title="Spectral Test",
            figsize=(15, 10),
            show_axis=True,
            colormap="viridis",
            show_colorbar=True,
            filename_prefix="SpectralTool",
            dpi=500,
            bbox_inches="tight",
            grid=False,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "Spectral Test",
        (15, 10),
        True,
        "viridis",
        True,
        "SpectralTool",
        500,
        "tight",
        False,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_spectral_profile_calculator):
    with patch(
        "fezrs.tools.spectral_profile.spectral_profile_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_spectral_profile_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (10, 10),
        True,
        "gray",
        False,
        "Tool_output",
        1000,
        "tight",
        True,
    )
    assert result == "executed"


def test_customize_export_file_method_exists(mock_spectral_profile_calculator):
    mock_spectral_profile_calculator._customize_export_file(MagicMock())
