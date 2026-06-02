import numpy as np
import pytest

from fezrs.tools.filters.gaussian_calculator import GaussianCalculator


@pytest.fixture
def calculator():
    obj = GaussianCalculator.__new__(GaussianCalculator)

    obj.metadata_bands = {
        "tif": {
            "image_skimage": np.array(
                [
                    [10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90],
                ],
                dtype=np.uint8,
            )
        }
    }

    obj._output = None

    return obj


def test_process_returns_numpy_array(calculator):
    result = calculator.process()

    assert isinstance(result, np.ndarray)


def test_process_preserves_shape(calculator):
    result = calculator.process()

    assert result.shape == calculator.metadata_bands["tif"]["image_skimage"].shape


def test_process_sets_output(calculator):
    result = calculator.process()

    assert calculator._output is result


def test_process_changes_image_values(calculator):
    result = calculator.process()

    assert not np.array_equal(
        result,
        calculator.metadata_bands["tif"]["image_skimage"],
    )


def test_execute_returns_self():
    calculator = GaussianCalculator.__new__(GaussianCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the GaussianCalculator
# if __name__ == "__main__":
#     tif_path = Path.cwd() / "data/IMG.tif"

#     calculator = GaussianCalculator(tif_path=tif_path).execute(
#         output_path="./", title="Gaussian output"
#     )
