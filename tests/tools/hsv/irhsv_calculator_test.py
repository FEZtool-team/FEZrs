import numpy as np
import pytest

from fezrs.tools.hsv.irhsv_calculator import IRHSVCalculator


@pytest.fixture
def normalized_bands():
    return {
        "red": np.array(
            [
                [0.2, 0.4],
                [0.6, 0.8],
            ]
        ),
        "swir1": np.array(
            [
                [0.1, 0.3],
                [0.5, 0.7],
            ]
        ),
        "swir2": np.array(
            [
                [0.0, 0.2],
                [0.4, 0.6],
            ]
        ),
    }


@pytest.fixture
def calculator(normalized_bands):
    obj = IRHSVCalculator.__new__(IRHSVCalculator)

    obj.normalized_bands = normalized_bands
    obj.selected_channel = "irhsv"
    obj._output = None

    return obj


def test_process_returns_irhsv_array(calculator):
    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 3)


def test_process_sets_output(calculator):
    result = calculator.process()

    assert calculator._output is result


def test_process_returns_irhue_channel(normalized_bands):
    calculator = IRHSVCalculator.__new__(IRHSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "irhue"

    result = calculator.process()

    assert result.shape == (2, 2)


def test_process_returns_irsaturation_channel(normalized_bands):
    calculator = IRHSVCalculator.__new__(IRHSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "irsaturation"

    result = calculator.process()

    assert result.shape == (2, 2)


def test_process_returns_irvalue_channel(normalized_bands):
    calculator = IRHSVCalculator.__new__(IRHSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "irvalue"

    result = calculator.process()

    assert result.shape == (2, 2)


def test_process_invalid_channel_raises_keyerror(normalized_bands):
    calculator = IRHSVCalculator.__new__(IRHSVCalculator)

    calculator.normalized_bands = normalized_bands
    calculator.selected_channel = "invalid"

    with pytest.raises(KeyError):
        calculator.process()


def test_validate_does_not_raise(calculator):
    calculator._validate()


def test_execute_returns_self():
    calculator = IRHSVCalculator.__new__(IRHSVCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the IRHSVCalculator
# if __name__ == "__main__":
#     red_path = Path.cwd() / "data/Red.tif"
#     swir1_path = Path.cwd() / "data/SWIR1.tif"
#     swir2_path = Path.cwd() / "data/SWIR2.tif"

#     calculator = IRHSVCalculator(
#         red_path=red_path, swir1_path=swir1_path, swir2_path=swir2_path, channel="irhue"
#     ).execute(output_path="./", title="IRHSV output")
