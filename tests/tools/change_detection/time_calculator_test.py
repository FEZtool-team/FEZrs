import numpy as np
import pytest

from fezrs.tools.change_detection.time_calculator import TimeCalculator


@pytest.fixture
def calculator():
    obj = TimeCalculator.__new__(TimeCalculator)

    obj.time_bands = {
        "nir": {"image_skimage": np.array([[1, 2], [3, 4]])},
        "before_nir": {"image_skimage": np.array([[5, 6], [7, 8]])},
    }

    obj._output = None

    return obj


def test_process_after(calculator):
    calculator.selectedTime = "after"

    result = calculator.process()

    np.testing.assert_array_equal(
        result,
        calculator.time_bands["nir"]["image_skimage"],
    )


def test_process_before(calculator):
    calculator.selectedTime = "before"

    result = calculator.process()

    np.testing.assert_array_equal(
        result,
        calculator.time_bands["before_nir"]["image_skimage"],
    )


def test_process_unknown_time_falls_back_to_after(calculator):
    calculator.selectedTime = "unknown"

    result = calculator.process()

    np.testing.assert_array_equal(
        result,
        calculator.time_bands["nir"]["image_skimage"],
    )


def test_process_sets_output(calculator):
    calculator.selectedTime = "after"

    result = calculator.process()

    assert calculator._output is result


def test_execute_returns_self():
    calculator = TimeCalculator.__new__(TimeCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the TimeCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/Change Detection/After/B4.tif"
#     swir1_path = Path.cwd() / "data/Change Detection/After/B5.tif"
#     swir2_path = Path.cwd() / "data/Change Detection/After/B7.tif"

#     before_nir_path = Path.cwd() / "data/Change Detection/Before/B4.tif"
#     before_swir1_path = Path.cwd() / "data/Change Detection/Before/B5.tif"
#     before_swir2_path = Path.cwd() / "data/Change Detection/Before/B7.tif"

#     calculator = TimeCalculator(
#         nir_path=nir_path,
#         before_nir_path=before_nir_path,
#         time="before",
#     ).execute("./", title="Test CD time")
