import numpy as np
import pytest

from fezrs.tools.clustering.kmeans_calculator import KMeansCalculator


@pytest.fixture
def calculator():
    obj = KMeansCalculator.__new__(KMeansCalculator)

    obj.n_clusters = 2
    obj.random_state = 0

    obj.nir_band = np.array(
        [
            [1.0, 1.0],
            [10.0, 10.0],
        ]
    )

    obj.metadata_bands = {
        "nir": {
            "width": 2,
            "height": 2,
        }
    }

    obj._output = None

    return obj


def test_validate_rejects_non_integer_n_clusters(calculator):
    calculator.n_clusters = "2"

    with pytest.raises(TypeError):
        calculator._validate()


def test_validate_rejects_small_n_clusters(calculator):
    calculator.n_clusters = 1

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_invalid_random_state(calculator):
    calculator.random_state = "seed"

    with pytest.raises(TypeError):
        calculator._validate()


def test_validate_rejects_non_ndarray_nir_band(calculator):
    calculator.nir_band = [[1, 2], [3, 4]]

    with pytest.raises(TypeError):
        calculator._validate()


def test_validate_rejects_non_2d_nir_band(calculator):
    calculator.nir_band = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_missing_metadata(calculator):
    calculator.metadata_bands = {}

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_invalid_width(calculator):
    calculator.metadata_bands["nir"]["width"] = 0

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_rejects_invalid_height(calculator):
    calculator.metadata_bands["nir"]["height"] = 0

    with pytest.raises(ValueError):
        calculator._validate()


def test_validate_accepts_valid_configuration(calculator):
    calculator._validate()


def test_process_returns_numpy_array(calculator):
    result = calculator.process()

    assert isinstance(result, np.ndarray)


def test_process_preserves_image_shape(calculator):
    result = calculator.process()

    assert result.shape == calculator.nir_band.shape


def test_process_sets_output(calculator):
    result = calculator.process()

    assert calculator._output is result


def test_execute_returns_self():
    calculator = KMeansCalculator.__new__(KMeansCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1.0]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the KMeansCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = KMeansCalculator(
#         nir_path=nir_path, n_clusters=4, random_state=0
#     ).execute(output_path="./", title="K-Means output")
