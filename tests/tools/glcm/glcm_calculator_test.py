import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from skimage.feature import graycomatrix, graycoprops

from fezrs.tools.glcm.glcm_calculator import (
    GLCMCalculator,
    quantize_to_gray_levels,
)


@pytest.fixture
def calculator():
    obj = GLCMCalculator.__new__(GLCMCalculator)

    image = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ],
        dtype=np.uint8,
    )

    obj.metadata_bands = {
        "nir": {
            "width": 4,
            "height": 4,
            "image_skimage": image,
        }
    }

    obj.result = np.empty((4, 4))
    obj.nir_image = image
    obj.window_size = 3
    obj.property = "contrast"
    obj._output = None

    return obj


def test_process_sets_output(calculator):
    calculator.process()

    assert calculator._output is calculator.result


def test_process_returns_expected_shape(calculator):
    calculator.process()

    assert calculator._output.shape == (4, 4)


def test_process_output_contains_finite_values(calculator):
    calculator.process()

    assert np.isfinite(calculator._output).all()


@pytest.mark.parametrize(
    "property_name",
    [
        "contrast",
        "ASM",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ],
)
def test_process_supports_all_properties(calculator, property_name):
    calculator.property = property_name

    calculator.process()

    assert calculator._output.shape == (4, 4)


def test_validate_accepts_valid_inputs(calculator):
    calculator.window_size = 3
    calculator.property = "contrast"
    calculator._validate()


@pytest.mark.parametrize("window_size", [3, 5, 7])
def test_validate_accepts_odd_window_size(calculator, window_size):
    calculator.window_size = window_size
    calculator._validate()


@pytest.mark.parametrize("window_size", [1, 2, 4, 0, -1])
def test_validate_rejects_invalid_window_size(calculator, window_size):
    calculator.window_size = window_size
    with pytest.raises(ValueError, match="window_size"):
        calculator._validate()


def test_validate_rejects_non_integer_window_size(calculator):
    calculator.window_size = 3.0
    with pytest.raises(ValueError, match="window_size must be an int"):
        calculator._validate()


def test_validate_rejects_invalid_property(calculator):
    calculator.property = "not_a_property"
    with pytest.raises(ValueError, match="Invalid GLCM property"):
        calculator._validate()


def test_validate_rejects_missing_input_path(calculator, tmp_path):
    calculator.files_handler = MagicMock()
    calculator.files_handler.band_paths = {
        "nir": str(tmp_path / "does_not_exist.tif"),
    }

    with pytest.raises(FileNotFoundError):
        calculator._validate()


def test_validate_accepts_existing_input_path(calculator, tmp_path):
    nir_path = tmp_path / "nir.tif"
    nir_path.write_bytes(b"placeholder")
    calculator.files_handler = MagicMock()
    calculator.files_handler.band_paths = {"nir": str(nir_path)}

    calculator._validate()


def test_init_rejects_nonexistent_input_path():
    with pytest.raises(FileNotFoundError):
        GLCMCalculator(
            nir_path="/this/path/does/not/exist.tif",
            window_size=3,
            propery="contrast",
        )


def test_process_rejects_invalid_window_size(calculator):
    calculator.window_size = 2
    with pytest.raises(ValueError, match="window_size"):
        calculator.process()


def test_execute_returns_self():
    calculator = GLCMCalculator.__new__(GLCMCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


HEIGHT = 4
WIDTH = 7
WINDOW_SIZE = 3

# Asymmetric 4x7 raster so a row/column swap cannot match the reference.
NONSQUARE_IMAGE = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7],
        [10, 20, 30, 40, 50, 60, 70],
        [3, 1, 4, 1, 5, 9, 2],
        [8, 8, 8, 1, 2, 3, 4],
    ],
    dtype=np.uint8,
)


def _expected_glcm_value(image, row, col, window_size, property_name):
    window = image[row : row + window_size, col : col + window_size]
    glcm = graycomatrix(window, [1], [0], normed=True, symmetric=True)
    return graycoprops(glcm, property_name)[0][0]


@pytest.fixture
def nonsquare_calculator():
    """
    4x7 GLCMCalculator built through __init__ so allocation uses (height, width).
    """
    fake_metadata = {
        "nir": {
            "width": WIDTH,
            "height": HEIGHT,
            "image_skimage": NONSQUARE_IMAGE,
        }
    }

    fake_files_handler = MagicMock()
    fake_files_handler.get_metadata_bands.return_value = fake_metadata

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None

    with patch(
        "fezrs.tools.glcm.glcm_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = GLCMCalculator(
            nir_path="dummy_nir.tif",
            window_size=WINDOW_SIZE,
            propery="contrast",
        )

    return calculator


def test_nonsquare_init_allocates_height_width(nonsquare_calculator):
    assert nonsquare_calculator.result.shape == (HEIGHT, WIDTH)
    assert nonsquare_calculator.nir_image.shape == (HEIGHT, WIDTH)


def test_nonsquare_output_shape(nonsquare_calculator):
    nonsquare_calculator.process()

    assert nonsquare_calculator._output.shape == (HEIGHT, WIDTH)


def test_nonsquare_interior_matches_skimage(nonsquare_calculator):
    """
    Interior pixels have a full window and must match graycomatrix/graycoprops.
    """
    nonsquare_calculator.process()

    interior_pixels = ((0, 0), (0, 2), (1, 4))
    for row, col in interior_pixels:
        window = NONSQUARE_IMAGE[
            row : row + WINDOW_SIZE, col : col + WINDOW_SIZE
        ]
        assert window.shape == (WINDOW_SIZE, WINDOW_SIZE)

        expected = _expected_glcm_value(
            NONSQUARE_IMAGE, row, col, WINDOW_SIZE, "contrast"
        )
        np.testing.assert_allclose(
            nonsquare_calculator._output[row, col],
            expected,
        )


def test_nonsquare_top_left_uses_full_window(nonsquare_calculator):
    """
    The window is anchored at (row, col) and extends down/right, so the
    top and left borders still have a full window_size x window_size block.
    """
    nonsquare_calculator.process()

    top_left_pixels = ((0, 0), (0, 3), (1, 0))
    for row, col in top_left_pixels:
        window = NONSQUARE_IMAGE[
            row : row + WINDOW_SIZE, col : col + WINDOW_SIZE
        ]
        assert window.shape == (WINDOW_SIZE, WINDOW_SIZE)
        expected = _expected_glcm_value(
            NONSQUARE_IMAGE, row, col, WINDOW_SIZE, "contrast"
        )
        np.testing.assert_allclose(
            nonsquare_calculator._output[row, col],
            expected,
        )


def test_nonsquare_border_uses_truncated_window(nonsquare_calculator):
    """
    Near the right and bottom edges the window is clipped to the image.
    Those pixels are still computed, not left uninitialized.
    """
    nonsquare_calculator.process()

    border_pixels = (
        (0, WIDTH - 1),
        (HEIGHT - 1, 0),
        (HEIGHT - 1, WIDTH - 1),
        (1, WIDTH - 2),
    )
    for row, col in border_pixels:
        window = NONSQUARE_IMAGE[
            row : row + WINDOW_SIZE, col : col + WINDOW_SIZE
        ]
        assert window.shape[0] < WINDOW_SIZE or window.shape[1] < WINDOW_SIZE

        expected = _expected_glcm_value(
            NONSQUARE_IMAGE, row, col, WINDOW_SIZE, "contrast"
        )
        np.testing.assert_allclose(
            nonsquare_calculator._output[row, col],
            expected,
        )
        assert np.isfinite(nonsquare_calculator._output[row, col])


def test_quantize_preserves_order_for_16bit_values():
    image = np.array(
        [[23, 1000, 3311], [2000, 6200, 255]],
        dtype=np.int16,
    )
    quantized = quantize_to_gray_levels(image, levels=256)

    assert quantized.dtype == np.uint8
    assert quantized.min() == 0
    assert quantized.max() == 255
    # Gray-level ordering is preserved (unlike a wraparound uint8 cast).
    assert quantized[0, 1] < quantized[1, 0]
    assert quantized[1, 0] < quantized[0, 2]
    correlation = np.corrcoef(image.ravel().astype(float), quantized.ravel().astype(float))[0, 1]
    assert correlation > 0.99
    wrapped = np.array(image, dtype="uint8")
    wrap_corr = np.corrcoef(image.ravel().astype(float), wrapped.ravel().astype(float))[0, 1]
    assert wrap_corr < 0.5


def test_quantize_leaves_in_range_uint8_unchanged():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    quantized = quantize_to_gray_levels(image, levels=256)
    np.testing.assert_array_equal(quantized, image)


def test_init_accepts_property_alias():
    fake_metadata = {
        "nir": {
            "width": 4,
            "height": 4,
            "image_skimage": np.arange(16, dtype=np.uint8).reshape(4, 4),
        }
    }
    fake_files_handler = MagicMock()
    fake_files_handler.get_metadata_bands.return_value = fake_metadata

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None

    with patch(
        "fezrs.tools.glcm.glcm_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = GLCMCalculator(
            nir_path="dummy_nir.tif",
            window_size=3,
            property="energy",
        )

    assert calculator.property == "energy"


def test_quantize_constant_image_is_zeros():
    image = np.full((3, 3), 512, dtype=np.int16)
    quantized = quantize_to_gray_levels(image, levels=256)
    np.testing.assert_array_equal(quantized, np.zeros((3, 3), dtype=np.uint8))


def test_quantize_rejects_invalid_levels():
    with pytest.raises(ValueError, match="levels"):
        quantize_to_gray_levels(np.ones((2, 2)), levels=1)


def test_process_does_not_print_by_default(calculator, capsys):
    calculator.process()
    captured = capsys.readouterr()
    assert captured.out == ""


# NOTE - These block code for integration test the GLCMCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = GLCMCalculator(
#         nir_path=nir_path, window_size=3, propery="ASM"
#     ).execute(output_path="./", title="GLCM output")
