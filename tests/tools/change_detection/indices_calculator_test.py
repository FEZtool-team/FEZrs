from pathlib import Path
import numpy as np
import pytest

from fezrs.tools.change_detection.indices_calculator import IndicesCalculator


@pytest.fixture
def calculator():
    obj = IndicesCalculator.__new__(IndicesCalculator)

    obj.time_bands = {
        "nir": {"image_skimage": np.array([[0.8, 0.6]])},
        "swir2": {"image_skimage": np.array([[0.2, 0.4]])},
        "before_nir": {"image_skimage": np.array([[0.9, 0.7]])},
        "before_swir2": {"image_skimage": np.array([[0.1, 0.3]])},
    }

    obj._output = None

    return obj


def test_process_after(calculator):
    calculator.selectedTime = "after"

    result = calculator.process()

    expected = (
        calculator.time_bands["nir"]["image_skimage"]
        - calculator.time_bands["swir2"]["image_skimage"]
    ) / (
        calculator.time_bands["nir"]["image_skimage"]
        + calculator.time_bands["swir2"]["image_skimage"]
    )

    np.testing.assert_allclose(result, expected)


def test_process_before(calculator):
    calculator.selectedTime = "before"

    result = calculator.process()

    expected = (
        calculator.time_bands["before_nir"]["image_skimage"]
        - calculator.time_bands["before_swir2"]["image_skimage"]
    ) / (
        calculator.time_bands["before_nir"]["image_skimage"]
        + calculator.time_bands["before_swir2"]["image_skimage"]
    )

    np.testing.assert_allclose(result, expected)


def test_process_unknown_time_falls_back_to_after(calculator):
    calculator.selectedTime = "unknown"

    result = calculator.process()

    expected = (
        calculator.time_bands["nir"]["image_skimage"]
        - calculator.time_bands["swir2"]["image_skimage"]
    ) / (
        calculator.time_bands["nir"]["image_skimage"]
        + calculator.time_bands["swir2"]["image_skimage"]
    )

    np.testing.assert_allclose(result, expected)


def test_process_sets_output(calculator):
    calculator.selectedTime = "after"

    result = calculator.process()

    assert calculator._output is result


def test_execute_returns_self():
    calculator = IndicesCalculator.__new__(IndicesCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1.0]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the IndicesCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/Change Detection/After/B4.tif"
#     swir1_path = Path.cwd() / "data/Change Detection/After/B5.tif"
#     swir2_path = Path.cwd() / "data/Change Detection/After/B7.tif"

#     before_nir_path = Path.cwd() / "data/Change Detection/Before/B4.tif"
#     before_swir1_path = Path.cwd() / "data/Change Detection/Before/B5.tif"
#     before_swir2_path = Path.cwd() / "data/Change Detection/Before/B7.tif"

#     calculator = IndicesCalculator(
#         nir_path=nir_path,
#         swir2_path=swir2_path,
#         before_nir_path=before_nir_path,
#         before_swir2_path=before_swir2_path,
#         time="after",
#     ).execute("./", title="Test CD time")
