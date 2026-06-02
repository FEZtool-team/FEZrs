import numpy as np
import pytest

from fezrs.tools.filters.laplacian_calculator import LaplacianCalculator


class DummyFileHandler:
    def __init__(self, band):
        self.bands = {"tif": band}


@pytest.fixture
def calculator():
    obj = LaplacianCalculator.__new__(LaplacianCalculator)

    obj.kernel_size = 3

    obj.files_handler = DummyFileHandler(
        np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
            dtype=np.uint8,
        )
    )

    obj.metadata_bands = {
        "tif": {
            "width": 3,
            "height": 3,
            "image_skimage": np.array(
                [
                    [10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90],
                ],
                dtype=np.uint8,
            ),
        }
    }

    obj._output = None

    return obj


def test_validate_rejects_non_integer_kernel_size(calculator):
    calculator.kernel_size = "3"

    with pytest.raises(TypeError):
        calculator._validate()


def test_validate_rejects_non_positive_kernel_size(calculator):
    calculator.kernel_size = 0

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_even_kernel_size(calculator):
    calculator.kernel_size = 4

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_missing_tif_band(calculator):
    calculator.files_handler.bands["tif"] = None

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_non_ndarray_tif_band(calculator):
    calculator.files_handler.bands["tif"] = [[1, 2], [3, 4]]

    with pytest.raises(TypeError):
        calculator._validate()


def test_validate_rejects_non_2d_tif_band(calculator):
    calculator.files_handler.bands["tif"] = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_missing_metadata(calculator):
    calculator.metadata_bands = {}

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_invalid_width(calculator):
    calculator.metadata_bands["tif"]["width"] = 0

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_invalid_height(calculator):
    calculator.metadata_bands["tif"]["height"] = 0

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_accepts_valid_configuration(calculator):
    calculator._validate()


def test_process_returns_numpy_array(calculator):
    result = calculator.process()

    assert isinstance(result, np.ndarray)


def test_process_preserves_shape(calculator):
    result = calculator.process()

    assert result.shape == calculator.metadata_bands["tif"]["image_skimage"].shape


def test_process_sets_output(calculator):
    result = calculator.process()

    assert calculator._output is result


def test_execute_returns_self():
    calculator = LaplacianCalculator.__new__(LaplacianCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the LaplacianCalculator
# if __name__ == "__main__":
#     tif_path = Path.cwd() / "data/IMG.tif"

#     calculator = LaplacianCalculator(tif_path=tif_path, kernel_size=7).execute(
#         output_path="./", title="Laplacian output"
#     )
