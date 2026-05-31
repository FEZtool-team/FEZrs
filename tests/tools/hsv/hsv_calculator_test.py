import numpy as np
import pytest

from fezrs.tools.hsv.hsv_calculator import HSVCalculator


@pytest.fixture
def normalized_bands():
    return {
        "nir": np.array(
            [
                [0.2, 0.4],
                [0.6, 0.8],
            ]
        ),
        "green": np.array(
            [
                [0.1, 0.3],
                [0.5, 0.7],
            ]
        ),
        "blue": np.array(
            [
                [0.0, 0.2],
                [0.4, 0.6],
            ]
        ),
    }


@pytest.fixture
def calculator(normalized_bands):
    obj = HSVCalculator.__new__(HSVCalculator)

    obj.normalized_bands = normalized_bands
    obj.selected_channel = "hsv"
    obj._output = None

    return obj


def test_process_returns_hsv_array(calculator):
    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 3)


def test_process_sets_output(calculator):
    result = calculator.process()

    assert calculator._output is result


def test_process_returns_hue_channel(normalized_bands):
    calculator = HSVCalculator.__new__(HSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "hue"

    result = calculator.process()

    assert result.shape == (2, 2)


def test_process_returns_saturation_channel(normalized_bands):
    calculator = HSVCalculator.__new__(HSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "saturation"

    result = calculator.process()

    assert result.shape == (2, 2)


def test_process_returns_value_channel(normalized_bands):
    calculator = HSVCalculator.__new__(HSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "value"

    result = calculator.process()

    assert result.shape == (2, 2)


def test_process_invalid_channel_raises_keyerror(normalized_bands):
    calculator = HSVCalculator.__new__(HSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "invalid"

    with pytest.raises(KeyError):
        calculator.process()


def test_validate_does_not_raise(calculator):
    calculator._validate()


def test_execute_returns_self():
    calculator = HSVCalculator.__new__(HSVCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the HSVCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"
#     blue_path = Path.cwd() / "data/Blue.tif"
#     green_path = Path.cwd() / "data/Green.tif"

#     calculator = HSVCalculator(
#         blue_path=blue_path, green_path=green_path, nir_path=nir_path, channel="hsv"
#     ).execute(output_path="./", title="HSV output")
