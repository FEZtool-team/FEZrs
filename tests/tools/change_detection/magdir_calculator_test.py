import numpy as np
import pytest

from fezrs.tools.change_detection.magdir_calculator import MagDirCalculator


@pytest.fixture
def calculator():
    obj = MagDirCalculator.__new__(MagDirCalculator)

    obj.time_bands = {
        "nir": {"image_skimage": np.array([[1.0, 3.0], [1.0, 3.0]])},
        "swir1": {"image_skimage": np.array([[1.0, 1.0], [3.0, 3.0]])},
        "before_nir": {"image_skimage": np.array([[2.0, 2.0], [2.0, 2.0]])},
        "before_swir1": {"image_skimage": np.array([[2.0, 2.0], [2.0, 2.0]])},
    }

    obj._output = None

    return obj


def test_process_magnitude(calculator):
    calculator.select = "magnitude"

    result = calculator.process()

    expected = np.array(
        [
            [np.sqrt(2), np.sqrt(2)],
            [np.sqrt(2), np.sqrt(2)],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_process_direction(calculator):
    calculator.select = "direction"

    result = calculator.process()

    expected = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_process_unknown_selection_falls_back_to_direction(calculator):
    calculator.select = "unknown"

    result = calculator.process()

    expected = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_process_sets_output(calculator):
    calculator.select = "magnitude"

    result = calculator.process()

    assert calculator._output is result


def test_execute_returns_self():
    calculator = MagDirCalculator.__new__(MagDirCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1.0]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the MagDirCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/Change Detection/After/B4.tif"
#     swir1_path = Path.cwd() / "data/Change Detection/After/B5.tif"

#     before_nir_path = Path.cwd() / "data/Change Detection/Before/B4.tif"
#     before_swir1_path = Path.cwd() / "data/Change Detection/Before/B5.tif"

#     calculator = MagDirCalculator(
#         nir_path=nir_path,
#         swir1_path=swir1_path,
#         before_nir_path=before_nir_path,
#         before_swir1_path=before_swir1_path,
#         selecte="magnitude",
#     ).execute(
#         "./", title="MagDir CD time", colormap=None, show_colorbar=True, show_axis=False
#     )
