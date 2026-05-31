import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.import_tools.landsat8_calculator import Landsat8_Calculator


@pytest.fixture
def mock_landsat8_calculator():
    fake_bands = {
        "red": np.random.rand(10, 10),
        "green": np.random.rand(10, 10),
        "blue": np.random.rand(10, 10),
        "nir": np.random.rand(10, 10),
        "swir1": np.random.rand(10, 10),
        "swir2": np.random.rand(10, 10),
    }

    fake_files_handler = MagicMock()
    fake_files_handler.bands = fake_bands

    def fake_get_normalized_bands(requested_bands):
        return {band: fake_bands[band] for band in requested_bands}

    fake_files_handler.get_normalized_bands.side_effect = fake_get_normalized_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.exportType = kwargs.get("exportType", None)

    with patch(
        "fezrs.tools.import_tools.landsat8_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = Landsat8_Calculator(
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
            blue_path="dummy_blue.tif",
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
            swir2_path="dummy_swir2.tif",
            exportType=None,
        )

    return calculator


def test_initialization_with_default_export(mock_landsat8_calculator):
    assert mock_landsat8_calculator.exportType is None
    assert mock_landsat8_calculator._output is None


def test_initialization_with_rgb_export(mock_landsat8_calculator):
    def fake_init(self, *args, **kwargs):
        self.files_handler = mock_landsat8_calculator.files_handler
        self._output = None
        self.exportType = "rgb"

    with patch(
        "fezrs.tools.import_tools.landsat8_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = Landsat8_Calculator(
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
            blue_path="dummy_blue.tif",
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
            swir2_path="dummy_swir2.tif",
            exportType="rgb",
        )

    assert calculator.exportType == "rgb"


def test_initialization_with_infrared_export(mock_landsat8_calculator):
    def fake_init(self, *args, **kwargs):
        self.files_handler = mock_landsat8_calculator.files_handler
        self._output = None
        self.exportType = "infrared"

    with patch(
        "fezrs.tools.import_tools.landsat8_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = Landsat8_Calculator(
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
            blue_path="dummy_blue.tif",
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
            swir2_path="dummy_swir2.tif",
            exportType="infrared",
        )

    assert calculator.exportType == "infrared"


def test_validate_method_exists(mock_landsat8_calculator):
    mock_landsat8_calculator._validate()


def test_process_with_export_type_none(mock_landsat8_calculator):
    mock_landsat8_calculator.exportType = None

    mock_landsat8_calculator.process()

    assert mock_landsat8_calculator._output is not None
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)

    expected_stack = np.stack(
        [
            mock_landsat8_calculator.files_handler.bands["red"],
            mock_landsat8_calculator.files_handler.bands["green"],
            mock_landsat8_calculator.files_handler.bands["blue"],
        ],
        axis=2,
    )

    assert np.array_equal(mock_landsat8_calculator._output, expected_stack)


def test_process_with_export_type_rgb(mock_landsat8_calculator):
    mock_landsat8_calculator.exportType = "rgb"

    mock_landsat8_calculator.process()

    assert mock_landsat8_calculator._output is not None
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)

    expected_stack = np.stack(
        [
            mock_landsat8_calculator.files_handler.bands["red"],
            mock_landsat8_calculator.files_handler.bands["green"],
            mock_landsat8_calculator.files_handler.bands["blue"],
        ],
        axis=2,
    )

    assert np.array_equal(mock_landsat8_calculator._output, expected_stack)

    mock_landsat8_calculator.files_handler.get_normalized_bands.assert_called_once_with(
        requested_bands=["red", "green", "blue"]
    )


def test_process_with_export_type_infrared(mock_landsat8_calculator):
    mock_landsat8_calculator.exportType = "infrared"

    mock_landsat8_calculator.process()

    assert mock_landsat8_calculator._output is not None
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)

    expected_stack = np.stack(
        [
            mock_landsat8_calculator.files_handler.bands["swir2"],
            mock_landsat8_calculator.files_handler.bands["swir1"],
            mock_landsat8_calculator.files_handler.bands["nir"],
        ],
        axis=2,
    )

    assert np.array_equal(mock_landsat8_calculator._output, expected_stack)

    mock_landsat8_calculator.files_handler.get_normalized_bands.assert_called_once_with(
        requested_bands=["swir2", "swir1", "nir"]
    )


def test_process_with_unknown_export_type(mock_landsat8_calculator):
    mock_landsat8_calculator.exportType = "unknown"

    mock_landsat8_calculator.process()

    assert mock_landsat8_calculator._output is not None
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)

    expected_stack = np.stack(
        [
            mock_landsat8_calculator.files_handler.bands["red"],
            mock_landsat8_calculator.files_handler.bands["green"],
            mock_landsat8_calculator.files_handler.bands["blue"],
        ],
        axis=2,
    )

    assert np.array_equal(mock_landsat8_calculator._output, expected_stack)


def test_process_stores_output(mock_landsat8_calculator):
    mock_landsat8_calculator.exportType = None
    mock_landsat8_calculator.process()

    assert mock_landsat8_calculator._output is not None
    assert isinstance(mock_landsat8_calculator._output, np.ndarray)


def test_execute_calls_base_execute(mock_landsat8_calculator):
    with patch(
        "fezrs.tools.import_tools.landsat8_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_landsat8_calculator.execute(
            output_path="output/test.png",
            title="Landsat8 Test",
            figsize=(15, 10),
            show_axis=True,
            colormap="viridis",
            show_colorbar=True,
            filename_prefix="Landsat8Tool",
            dpi=300,
            bbox_inches="tight",
            grid=True,
            nrows=2,
            ncols=2,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "Landsat8 Test",
        (15, 10),
        True,
        "viridis",
        True,
        "Landsat8Tool",
        300,
        "tight",
        True,
        2,
        2,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_landsat8_calculator):
    with patch(
        "fezrs.tools.import_tools.landsat8_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_landsat8_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (10, 10),
        False,
        None,
        False,
        "Tool_output",
        500,
        "tight",
        False,
        None,
        None,
    )
    assert result == "executed"


def test_output_shape_for_different_export_types(mock_landsat8_calculator):
    mock_landsat8_calculator.exportType = None
    mock_landsat8_calculator.process()
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)

    mock_landsat8_calculator.exportType = "rgb"
    mock_landsat8_calculator.process()
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)

    mock_landsat8_calculator.exportType = "infrared"
    mock_landsat8_calculator.process()
    assert mock_landsat8_calculator._output.shape == (10, 10, 3)


def test_files_handler_bands_accessible(mock_landsat8_calculator):
    assert "red" in mock_landsat8_calculator.files_handler.bands
    assert "green" in mock_landsat8_calculator.files_handler.bands
    assert "blue" in mock_landsat8_calculator.files_handler.bands
    assert "nir" in mock_landsat8_calculator.files_handler.bands
    assert "swir1" in mock_landsat8_calculator.files_handler.bands
    assert "swir2" in mock_landsat8_calculator.files_handler.bands
