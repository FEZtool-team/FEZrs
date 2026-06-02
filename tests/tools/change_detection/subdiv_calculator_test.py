import numpy as np
import pytest

from fezrs.tools.change_detection.subdiv_calculator import SubDivCalculator


@pytest.fixture
def calculator():
    obj = SubDivCalculator.__new__(SubDivCalculator)

    obj.time_bands = {
        "nir": {"image_skimage": np.array([[2.0, 4.0]])},
        "before_nir": {"image_skimage": np.array([[8.0, 12.0]])},
    }

    obj._output = None

    return obj


def test_process_divide(calculator):
    calculator.operation = "divide"

    result = calculator.process()

    expected = np.array([[4.0, 3.0]])

    np.testing.assert_allclose(result, expected)


def test_process_subtract(calculator):
    calculator.operation = "subtract"

    result = calculator.process()

    expected = np.array([[6.0, 8.0]])

    np.testing.assert_array_equal(result, expected)


def test_process_unknown_operation_falls_back_to_subtract(calculator):
    calculator.operation = "unknown"

    result = calculator.process()

    expected = np.array([[6.0, 8.0]])

    np.testing.assert_array_equal(result, expected)


def test_process_sets_output(calculator):
    calculator.operation = "divide"

    result = calculator.process()

    assert calculator._output is result


def test_execute_returns_self():
    calculator = SubDivCalculator.__new__(SubDivCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1.0]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the SubDivCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/Change Detection/After/B4.tif"
#     swir1_path = Path.cwd() / "data/Change Detection/After/B5.tif"
#     swir2_path = Path.cwd() / "data/Change Detection/After/B7.tif"

#     before_nir_path = Path.cwd() / "data/Change Detection/Before/B4.tif"
#     before_swir1_path = Path.cwd() / "data/Change Detection/Before/B5.tif"
#     before_swir2_path = Path.cwd() / "data/Change Detection/Before/B7.tif"

#     calculator = SubDivCalculator(
#         nir_path=nir_path, before_nir_path=before_nir_path, operation="subtract"
#     ).execute("./", title="SubDiv CD time")
