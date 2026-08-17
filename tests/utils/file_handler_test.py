import pytest
from unittest import mock
import numpy as np

from fezrs.utils.file_handler import FileHandler, _load_image, _normalize


def test_load_image_none_path():
    assert _load_image(None) is None


@mock.patch("fezrs.utils.file_handler.os.path.exists", return_value=True)
@mock.patch(
    "fezrs.utils.file_handler.io.imread", return_value=np.array([[1, 2], [3, 4]])
)
def test_load_image_valid_path(mock_imread, mock_exists):
    result = _load_image("image.tif")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]], dtype=float))


@mock.patch("fezrs.utils.file_handler.os.path.exists", return_value=False)
def test_load_image_file_not_found(mock_exists):
    with pytest.raises(FileNotFoundError):
        _load_image("nonexistent.jpg")


def test_normalize_scales_to_unit_interval():
    image = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(_normalize(image), [[0.0, 1 / 3], [2 / 3, 1.0]])


def test_get_bands_returns_unscaled_values():
    handler = FileHandler.__new__(FileHandler)
    handler.bands = {
        "nir": np.array([[100.0, 200.0]]),
        "red": np.array([[10.0, 40.0]]),
        "blue": None,
    }
    handler.band_paths = {}
    handler.tif_paths = None

    bands = handler.get_bands(["nir", "red", "blue"])
    np.testing.assert_array_equal(bands["nir"], [[100.0, 200.0]])
    assert "blue" not in bands

    normalized = handler.get_normalized_bands(["nir"])
    np.testing.assert_allclose(normalized["nir"], [[0.0, 1.0]])
