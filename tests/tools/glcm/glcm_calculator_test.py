import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from skimage.feature import graycomatrix, graycoprops

from fezrs.tools.glcm.glcm_calculator import (
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    DEFAULT_LEVELS,
    GLCMCalculator,
    quantize_to_levels,
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
    obj.window_size = 3
    obj.property = "contrast"
    obj.levels = DEFAULT_LEVELS
    obj.distances = DEFAULT_DISTANCES
    obj.angles = DEFAULT_ANGLES
    obj.centered = True
    obj.nir_image = quantize_to_levels(image, obj.levels)
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


def _expected_glcm_value(
    image,
    row,
    col,
    window_size,
    property_name,
    levels=DEFAULT_LEVELS,
    distances=DEFAULT_DISTANCES,
    angles=DEFAULT_ANGLES,
    centered=True,
):
    """
    Reference value mirroring the calculator: quantize globally, optionally
    reflect-pad so the window is centered, then average the property over every
    distance/angle pair.
    """
    quantized = quantize_to_levels(image, levels)

    if centered:
        quantized = np.pad(quantized, window_size // 2, mode="reflect")

    window = quantized[row : row + window_size, col : col + window_size]
    glcm = graycomatrix(
        window,
        distances,
        angles,
        levels=levels,
        normed=True,
        symmetric=True,
    )
    return graycoprops(glcm, property_name).mean()


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
    Interior pixels must match graycomatrix/graycoprops run over the same
    quantized, centered window.
    """
    nonsquare_calculator.process()

    interior_pixels = ((1, 1), (1, 4), (2, 3))
    for row, col in interior_pixels:
        expected = _expected_glcm_value(
            NONSQUARE_IMAGE, row, col, WINDOW_SIZE, "contrast"
        )
        np.testing.assert_allclose(
            nonsquare_calculator._output[row, col],
            expected,
        )


def test_centered_window_gives_every_pixel_a_full_neighbourhood(
    nonsquare_calculator,
):
    """
    Under the default centered window, reflect padding means border pixels get a
    full window_size x window_size neighbourhood rather than a truncated one,
    and every output pixel is finite.
    """
    nonsquare_calculator.process()

    border_pixels = (
        (0, 0),
        (0, WIDTH - 1),
        (HEIGHT - 1, 0),
        (HEIGHT - 1, WIDTH - 1),
    )
    for row, col in border_pixels:
        expected = _expected_glcm_value(
            NONSQUARE_IMAGE, row, col, WINDOW_SIZE, "contrast"
        )
        np.testing.assert_allclose(
            nonsquare_calculator._output[row, col],
            expected,
        )

    assert np.isfinite(nonsquare_calculator._output).all()


def test_centered_window_is_not_offset_from_the_source_grid():
    """
    A centered window puts the texture response of a feature on the feature's
    own pixel. With the legacy anchored window the same response lands
    (window_size - 1) // 2 pixels up and to the left, which offsets the whole
    texture map relative to the source raster.
    """
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 200  # single bright pixel at the exact centre

    def build(centered):
        obj = GLCMCalculator.__new__(GLCMCalculator)
        obj.metadata_bands = {
            "nir": {"width": 9, "height": 9, "image_skimage": image}
        }
        obj.result = np.empty((9, 9))
        obj.window_size = 3
        obj.property = "contrast"
        obj.levels = DEFAULT_LEVELS
        obj.distances = DEFAULT_DISTANCES
        obj.angles = DEFAULT_ANGLES
        obj.centered = centered
        obj.nir_image = quantize_to_levels(image, obj.levels)
        obj._output = None
        obj.process()
        return obj._output

    centered = build(True)
    anchored = build(False)

    # The strongest contrast sits on the bright pixel itself when centered...
    assert np.unravel_index(np.argmax(centered), centered.shape) == (4, 4)
    # ...and is displaced up/left by one pixel under the legacy anchoring.
    assert np.unravel_index(np.argmax(anchored), anchored.shape) == (3, 3)


def test_legacy_anchored_window_still_available(nonsquare_calculator):
    """
    centered=False preserves the previous behaviour: the window is anchored at
    (row, col), extends down and to the right, and is clipped at the right and
    bottom edges.
    """
    nonsquare_calculator.centered = False
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
            NONSQUARE_IMAGE,
            row,
            col,
            WINDOW_SIZE,
            "contrast",
            centered=False,
        )
        np.testing.assert_allclose(
            nonsquare_calculator._output[row, col],
            expected,
        )
        assert np.isfinite(nonsquare_calculator._output[row, col])


# NOTE - These block code for integration test the GLCMCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = GLCMCalculator(
#         nir_path=nir_path, window_size=3, propery="ASM"
#     ).execute(output_path="./", title="GLCM output")


# --- Quantization (issue #38) -------------------------------------------------


def test_quantization_preserves_gray_level_ordering():
    """
    The previous np.array(..., dtype="uint8") wrapped modulo 256, so DN 3311
    became 239 and DN 6200 became 56 -- radiometrically adjacent pixels landed at
    opposite ends of the gray-level range. Quantization must be monotonic.
    """
    values = np.array([[23, 255, 256, 3311, 6200]], dtype=np.int16)

    quantized = quantize_to_levels(values, DEFAULT_LEVELS)

    assert np.all(np.diff(quantized[0]) >= 0)
    assert np.corrcoef(values.ravel(), quantized.ravel())[0, 1] > 0.99


def test_quantization_of_16bit_raster_is_not_a_wraparound_cast():
    rng = np.random.default_rng(0)
    band = rng.integers(23, 6200, size=(64, 64)).astype(np.int16)

    wrapped = np.array(band, dtype="uint8")
    quantized = quantize_to_levels(band, DEFAULT_LEVELS)

    assert np.corrcoef(band.ravel(), wrapped.ravel())[0, 1] < 0.5
    assert np.corrcoef(band.ravel(), quantized.ravel())[0, 1] > 0.99


def test_quantization_respects_levels():
    band = np.arange(0, 10000, dtype=np.int32).reshape(100, 100)

    for levels in (2, 16, 32, 64, 256):
        quantized = quantize_to_levels(band, levels)
        assert quantized.min() == 0
        assert quantized.max() == levels - 1


def test_quantization_of_constant_band_is_all_zero():
    band = np.full((8, 8), 4200, dtype=np.int16)

    quantized = quantize_to_levels(band, DEFAULT_LEVELS)

    assert quantized.dtype == np.uint8
    assert np.all(quantized == 0)


def test_quantization_is_global_not_per_window():
    """
    Two windows with the same local spread but different absolute levels must
    quantize differently, otherwise texture values are not comparable across the
    scene.
    """
    band = np.zeros((4, 8), dtype=np.int16)
    band[:, :4] = np.tile(np.array([100, 110], dtype=np.int16), (4, 2))
    band[:, 4:] = np.tile(np.array([5000, 5010], dtype=np.int16), (4, 2))

    quantized = quantize_to_levels(band, DEFAULT_LEVELS)

    assert quantized[:, :4].max() < quantized[:, 4:].min()


def test_validate_rejects_out_of_range_levels(calculator):
    for levels in (1, 0, -4, 257):
        calculator.levels = levels
        with pytest.raises(ValueError, match="levels"):
            calculator._validate()


def test_validate_rejects_non_integer_levels(calculator):
    calculator.levels = 64.0
    with pytest.raises(ValueError, match="levels must be an int"):
        calculator._validate()


# --- Orientation (issue #38) --------------------------------------------------


def test_angle_averaged_texture_is_rotation_invariant():
    """
    A striped pattern and its 90-degree rotation must yield the same texture
    under the default angle-averaged form.
    """
    stripes = np.tile(np.array([[0], [200]], dtype=np.uint8), (4, 8))
    rotated = np.rot90(stripes)

    def texture(image, angles):
        obj = GLCMCalculator.__new__(GLCMCalculator)
        height, width = image.shape
        obj.metadata_bands = {
            "nir": {"width": width, "height": height, "image_skimage": image}
        }
        obj.result = np.empty((height, width))
        obj.window_size = 3
        obj.property = "contrast"
        obj.levels = DEFAULT_LEVELS
        obj.distances = DEFAULT_DISTANCES
        obj.angles = angles
        obj.centered = True
        obj.nir_image = quantize_to_levels(image, obj.levels)
        obj._output = None
        obj.process()
        return float(np.mean(obj._output))

    averaged = (texture(stripes, DEFAULT_ANGLES), texture(rotated, DEFAULT_ANGLES))
    single = (texture(stripes, (0.0,)), texture(rotated, (0.0,)))

    assert averaged[0] == pytest.approx(averaged[1])
    # A single east-west orientation is direction dependent by construction.
    assert single[0] != pytest.approx(single[1])


def test_validate_rejects_empty_angles_and_distances(calculator):
    calculator.angles = ()
    with pytest.raises(ValueError, match="angles"):
        calculator._validate()

    calculator.angles = DEFAULT_ANGLES
    calculator.distances = ()
    with pytest.raises(ValueError, match="distances"):
        calculator._validate()


def test_validate_rejects_non_positive_distances(calculator):
    calculator.distances = (0,)
    with pytest.raises(ValueError, match="distances"):
        calculator._validate()


# --- Property spelling (issue #38 / #32) --------------------------------------


def _fake_base_init(self, *args, **kwargs):
    """Stand in for BaseTool.__init__ so no file is touched."""
    handler = MagicMock()
    handler.get_metadata_bands.return_value = {
        "nir": {
            "width": 4,
            "height": 4,
            "image_skimage": np.zeros((4, 4), dtype=np.uint8),
        }
    }
    self.files_handler = handler
    self._output = None


def test_property_keyword_is_accepted():
    with patch("fezrs.tools.glcm.glcm_calculator.BaseTool.__init__", _fake_base_init):
        calculator = GLCMCalculator(nir_path="dummy.tif", property="energy")

    assert calculator.property == "energy"


def test_deprecated_propery_keyword_still_works():
    with patch("fezrs.tools.glcm.glcm_calculator.BaseTool.__init__", _fake_base_init):
        calculator = GLCMCalculator(nir_path="dummy.tif", propery="energy")

    assert calculator.property == "energy"


def test_passing_both_property_spellings_raises():
    with patch("fezrs.tools.glcm.glcm_calculator.BaseTool.__init__", _fake_base_init):
        with pytest.raises(ValueError, match="not both"):
            GLCMCalculator(
                nir_path="dummy.tif", propery="energy", property="contrast"
            )


def test_property_defaults_to_contrast():
    with patch("fezrs.tools.glcm.glcm_calculator.BaseTool.__init__", _fake_base_init):
        calculator = GLCMCalculator(nir_path="dummy.tif")

    assert calculator.property == "contrast"


# --- Logging (issue #38) ------------------------------------------------------


def test_process_writes_nothing_to_stdout(calculator, capsys):
    """
    The per-row print wrote one line per raster row -- 998 lines for the bundled
    example, with no way to disable it.
    """
    calculator.process()

    assert capsys.readouterr().out == ""
