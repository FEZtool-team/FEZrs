"""
Non-interactive SVM training (issue #39).

Sample collection previously happened only through an OpenCV window, so the
classification path could not be exercised end to end: the existing tests all
mock the click interaction, and the classifier itself was never run against
data. These tests drive the real fit/predict path.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.svm.svm_calculator import SVMCalculator, display_is_available


HEIGHT = 12
WIDTH = 20


def _separable_scene():
    """
    Six bands over a scene whose left half is class 1 and right half class 2,
    with a clear spectral separation so the classifier has a right answer.
    """
    rng = np.random.default_rng(5)
    bands = []
    for offset in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        band = np.empty((HEIGHT, WIDTH), dtype=float)
        band[:, : WIDTH // 2] = 0.2 + offset + rng.normal(scale=0.01, size=(HEIGHT, WIDTH // 2))
        band[:, WIDTH // 2 :] = 0.8 + offset + rng.normal(scale=0.01, size=(HEIGHT, WIDTH // 2))
        bands.append(band)
    return bands


def _build(training_samples=None, transform=None, **kwargs):
    bands = _separable_scene()

    handler = MagicMock()
    handler.get_normalized_bands.return_value = {
        "red": bands[0],
        "green": bands[1],
        "blue": bands[2],
    }
    handler.get_metadata_bands.return_value = {
        "blue": {"height": HEIGHT, "width": WIDTH}
    }
    handler.get_images_collection.return_value = bands

    def fake_init(self, *args, **inner):
        self.files_handler = handler
        self._output = None

    with patch("fezrs.tools.svm.svm_calculator.BaseTool.__init__", fake_init):
        return SVMCalculator(
            red_path="red.tif",
            green_path="green.tif",
            blue_path="blue.tif",
            nir_path="nir.tif",
            swir1_path="swir1.tif",
            swir2_path="swir2.tif",
            training_samples=training_samples,
            transform=transform,
            **kwargs,
        )


LEFT = [(row, 2, 1) for row in range(4)]
RIGHT = [(row, WIDTH - 3, 2) for row in range(4)]
SAMPLES = LEFT + RIGHT


def test_classification_runs_without_a_gui():
    calculator = _build(training_samples=SAMPLES)

    output = calculator.process()

    assert output.shape == (HEIGHT, WIDTH)
    assert set(np.unique(output)) == {1, 2}


def test_classification_recovers_the_known_classes():
    """The scene is linearly separable, so a correct pipeline must classify it."""
    calculator = _build(training_samples=SAMPLES)

    output = calculator.process()

    assert np.all(output[:, : WIDTH // 2] == 1)
    assert np.all(output[:, WIDTH // 2 :] == 2)


def test_classification_is_reproducible():
    first = _build(training_samples=SAMPLES).process()
    second = _build(training_samples=SAMPLES).process()

    np.testing.assert_array_equal(first, second)


def test_training_samples_skip_the_gui_entirely():
    """No OpenCV entry point may be reached on the programmatic path."""
    calculator = _build(training_samples=SAMPLES)

    with (
        patch("fezrs.tools.svm.svm_calculator.cv2.namedWindow") as window,
        patch("fezrs.tools.svm.svm_calculator.cv2.imshow") as imshow,
        patch("fezrs.tools.svm.svm_calculator.cv2.waitKey") as wait,
    ):
        calculator.process()

    window.assert_not_called()
    imshow.assert_not_called()
    wait.assert_not_called()


# --- Map coordinates ----------------------------------------------------------


class _Affine:
    """
    Minimal stand-in for a rasterio Affine: 30 m pixels, origin at
    (316725, 4176795), north-up. Supports ~transform * (x, y).
    """

    def __init__(self, origin_x=316725.0, origin_y=4176795.0, pixel=30.0):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel = pixel

    def __invert__(self):
        return self

    def __rmul__(self, other):
        raise NotImplementedError

    def __mul__(self, coordinate):
        x, y = coordinate
        return ((x - self.origin_x) / self.pixel, (self.origin_y - y) / self.pixel)


def test_training_samples_accept_map_coordinates():
    """
    A training set digitised in a GIS carries easting/northing, not array
    indices, so pixel-only input keeps the tool out of a real mapping workflow.
    """
    transform = _Affine()

    def to_map(row, col, class_id):
        x = transform.origin_x + col * transform.pixel
        y = transform.origin_y - row * transform.pixel
        return (y, x, class_id)  # (northing, easting, class)

    map_samples = [to_map(*sample) for sample in SAMPLES]

    pixel_result = _build(training_samples=SAMPLES).process()
    map_result = _build(training_samples=map_samples, transform=transform).process()

    np.testing.assert_array_equal(pixel_result, map_result)


def test_sample_outside_the_raster_is_rejected():
    calculator = _build(training_samples=[(0, 0, 1), (HEIGHT + 50, 3, 2)])

    with pytest.raises(ValueError, match="outside the raster extent"):
        calculator.process()


# --- Validation ---------------------------------------------------------------


def test_training_samples_must_cover_two_classes():
    calculator = _build(training_samples=[(0, 1, 1), (1, 2, 1)])

    with pytest.raises(ValueError, match="at least two classes"):
        calculator.process()


def test_training_samples_must_not_be_empty():
    calculator = _build(training_samples=[])

    # An empty sequence falls back to the interactive path only if it is None;
    # an explicit empty list is a caller error.
    with pytest.raises(ValueError, match="must not be empty"):
        calculator._validate()


# --- Headless guard -----------------------------------------------------------


def test_headless_run_without_samples_raises_a_clear_error():
    """
    Without a display, cv2.imshow aborts the process at the Qt layer instead of
    raising, so a caller cannot degrade gracefully. The guard turns that into a
    catchable error naming the fix.
    """
    calculator = _build(training_samples=None)

    with patch(
        "fezrs.tools.svm.svm_calculator.display_is_available", return_value=False
    ):
        with pytest.raises(RuntimeError, match="needs training_samples"):
            calculator.process()


def test_display_detection_reads_the_environment(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert display_is_available() is False

    monkeypatch.setenv("DISPLAY", ":0")
    assert display_is_available() is True


# --- Accuracy assessment ------------------------------------------------------


def test_evaluate_reports_accuracy_and_kappa():
    """
    The paper presents SVM land-cover classification as a demonstrated result.
    Cohen's kappa is the standard accuracy measure in the remote sensing
    literature and there was no way to obtain one.
    """
    samples = [(row, col, 1) for row in range(6) for col in (1, 2, 3)]
    samples += [(row, col, 2) for row in range(6) for col in (WIDTH - 2, WIDTH - 3, WIDTH - 4)]

    calculator = _build(training_samples=samples, evaluate=True, random_state=0)
    calculator.process()

    assert calculator.accuracy_ == pytest.approx(1.0)
    assert calculator.kappa_ == pytest.approx(1.0)
    assert calculator.confusion_matrix_.shape == (2, 2)


def test_metrics_are_none_without_evaluation():
    calculator = _build(training_samples=SAMPLES)
    calculator.process()

    assert calculator.accuracy_ is None
    assert calculator.kappa_ is None
    assert calculator.confusion_matrix_ is None
