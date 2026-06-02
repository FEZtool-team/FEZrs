import numpy as np
import pytest

from fezrs.tools.change_detection.burn_calculator import BurnCalculator


@pytest.fixture
def burn_calculator():
    calculator = BurnCalculator.__new__(BurnCalculator)

    calculator.time_bands = {
        "nir": {"image_skimage": np.array([[0.1]])},
        "swir2": {"image_skimage": np.array([[0.9]])},
        "before_nir": {"image_skimage": np.array([[0.9]])},
        "before_swir2": {"image_skimage": np.array([[0.1]])},
    }

    calculator._output = None

    return calculator


def test_process_returns_boolean_array(burn_calculator):
    result = burn_calculator.process()

    assert result.dtype == bool


def test_process_sets_output(burn_calculator):
    result = burn_calculator.process()

    assert burn_calculator._output is result


def test_process_detects_burn_area(burn_calculator):
    result = burn_calculator.process()

    expected = np.array([[True]])

    np.testing.assert_array_equal(result, expected)


def test_process_returns_false_when_threshold_not_reached():
    calculator = BurnCalculator.__new__(BurnCalculator)

    calculator.time_bands = {
        "nir": {"image_skimage": np.array([[0.5]])},
        "swir2": {"image_skimage": np.array([[0.5]])},
        "before_nir": {"image_skimage": np.array([[0.5]])},
        "before_swir2": {"image_skimage": np.array([[0.5]])},
    }

    result = calculator.process()

    expected = np.array([[False]])

    np.testing.assert_array_equal(result, expected)


def test_execute_returns_self(monkeypatch):
    calculator = BurnCalculator.__new__(BurnCalculator)

    def fake_validate():
        pass

    def fake_process():
        calculator._output = np.array([[True]])

    def fake_export(*args, **kwargs):
        return "output.png"

    calculator._validate = fake_validate
    calculator.process = fake_process
    calculator._export_file = fake_export

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the BurnCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/Change Detection/After/B4.tif"
#     swir1_path = Path.cwd() / "data/Change Detection/After/B5.tif"
#     swir2_path = Path.cwd() / "data/Change Detection/After/B7.tif"

#     before_nir_path = Path.cwd() / "data/Change Detection/Before/B4.tif"
#     before_swir1_path = Path.cwd() / "data/Change Detection/Before/B5.tif"
#     before_swir2_path = Path.cwd() / "data/Change Detection/Before/B7.tif"

#     calculator = BurnCalculator(
#         nir_path=nir_path,
#         swir2_path=swir2_path,
#         before_nir_path=before_nir_path,
#         before_swir2_path=before_swir2_path,
#     ).execute("./", title="Burn CD time")
