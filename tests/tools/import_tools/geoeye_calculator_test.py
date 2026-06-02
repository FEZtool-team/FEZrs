import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.import_tools.geoeye_calculator import Geoeye_Calculator


@pytest.fixture
def mock_geoeye_calculator():
    fake_normalized_data = {"tif": np.random.rand(10, 10, 5)}

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_data

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.tif_normalized = fake_normalized_data
        self.level = kwargs.get("level", 0)

    with patch(
        "fezrs.tools.import_tools.geoeye_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = Geoeye_Calculator(
            tif_path="dummy_geoeye.tif",
            level=3,
        )

    return calculator


def test_initialization(mock_geoeye_calculator):
    assert mock_geoeye_calculator.tif_normalized is not None
    assert "tif" in mock_geoeye_calculator.tif_normalized
    assert mock_geoeye_calculator.level == 3
    assert mock_geoeye_calculator._output is None


def test_validate_with_valid_level(mock_geoeye_calculator):
    mock_geoeye_calculator.tif_normalized = {"tif": np.random.rand(10, 10, 5)}
    mock_geoeye_calculator.level = 2
    mock_geoeye_calculator._validate()


def test_validate_with_invalid_level_negative(mock_geoeye_calculator):
    mock_geoeye_calculator.tif_normalized = {"tif": np.random.rand(10, 10, 5)}
    mock_geoeye_calculator.level = -1
    with pytest.raises(
        ValueError, match="Invalid level -1. It must be between 0 and 4."
    ):
        mock_geoeye_calculator._validate()


def test_validate_with_invalid_level_too_high(mock_geoeye_calculator):
    mock_geoeye_calculator.tif_normalized = {"tif": np.random.rand(10, 10, 5)}
    mock_geoeye_calculator.level = 5
    with pytest.raises(
        ValueError, match="Invalid level 5. It must be between 0 and 4."
    ):
        mock_geoeye_calculator._validate()


def test_process_selects_correct_band(mock_geoeye_calculator):
    mock_geoeye_calculator.tif_normalized = {"tif": np.random.rand(10, 10, 5)}
    mock_geoeye_calculator.level = 2
    expected_band = mock_geoeye_calculator.tif_normalized["tif"][:, :, 2]

    mock_geoeye_calculator.process()

    assert np.array_equal(mock_geoeye_calculator.tif_normalize_level, expected_band)
    assert np.array_equal(mock_geoeye_calculator._output, expected_band)


def test_process_stores_output(mock_geoeye_calculator):
    mock_geoeye_calculator.tif_normalized = {"tif": np.random.rand(10, 10, 5)}
    mock_geoeye_calculator.level = 1

    mock_geoeye_calculator.process()

    assert mock_geoeye_calculator._output is not None
    assert mock_geoeye_calculator._output.shape == (10, 10)


def test_execute_calls_base_execute(mock_geoeye_calculator):
    with patch(
        "fezrs.tools.import_tools.geoeye_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_geoeye_calculator.execute(
            output_path="output/test.png",
            title="GeoEye Test",
            figsize=(15, 10),
            show_axis=True,
            colormap="viridis",
            show_colorbar=True,
            filename_prefix="GeoEyeTool",
            dpi=300,
            bbox_inches="tight",
            grid=True,
            nrows=2,
            ncols=2,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "GeoEye Test",
        (15, 10),
        True,
        "viridis",
        True,
        "GeoEyeTool",
        300,
        "tight",
        True,
        2,
        2,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_geoeye_calculator):
    with patch(
        "fezrs.tools.import_tools.geoeye_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_geoeye_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (10, 10),
        True,
        "gray",
        False,
        "Tool_output",
        500,
        "tight",
        False,
        None,
        None,
    )
    assert result == "executed"


def test_level_property_accessible(mock_geoeye_calculator):
    assert mock_geoeye_calculator.level == 3
    mock_geoeye_calculator.level = 1
    assert mock_geoeye_calculator.level == 1


def test_tif_normalized_structure(mock_geoeye_calculator):
    assert "tif" in mock_geoeye_calculator.tif_normalized
    assert isinstance(mock_geoeye_calculator.tif_normalized["tif"], np.ndarray)
    assert mock_geoeye_calculator.tif_normalized["tif"].ndim == 3


# if __name__ == "__main__":
#     tif_path = Path.cwd() / "data/Geoeye/geoeye.tif"

#     calculator = Geoeye_Calculator(tif_path=tif_path, level=3).execute(
#         "./", title="GeoEye"
#     )
