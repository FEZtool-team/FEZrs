import numpy as np
import pytest

from fezrs.tools.glcm.glcm_calculator import GLCMCalculator


@pytest.fixture
def calculator():
    obj = GLCMCalculator.__new__(GLCMCalculator)

    image = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ],
        dtype=np.uint8,
    )

    obj.metadata_bands = {
        "nir": {
            "width": 4,
            "height": 4,
            "image_skimage": image,
        }
    }

    obj.result = np.empty((4, 4))
    obj.nir_image = image
    obj.window_size = 2
    obj.property = "contrast"
    obj._output = None

    return obj


def test_process_sets_output(calculator):
    calculator.process()

    assert calculator._output is calculator.result


def test_process_returns_expected_shape(calculator):
    calculator.process()

    assert calculator._output.shape == (4, 4)


def test_process_output_contains_finite_values(calculator):
    calculator.process()

    assert np.isfinite(calculator._output).all()


@pytest.mark.parametrize(
    "property_name",
    [
        "contrast",
        "ASM",
        "dissimilarity",
        "homogeneity",
    ],
)
def test_process_supports_all_properties(calculator, property_name):
    calculator.property = property_name

    calculator.process()

    assert calculator._output.shape == (4, 4)


def test_validate_does_not_raise(calculator):
    calculator._validate()


def test_execute_returns_self():
    calculator = GLCMCalculator.__new__(GLCMCalculator)

    calculator._validate = lambda: None
    calculator.process = lambda: setattr(calculator, "_output", np.array([[1]]))
    calculator._export_file = lambda *args, **kwargs: "output.png"

    result = calculator.execute(".")

    assert result is calculator


# NOTE - These block code for integration test the GLCMCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = GLCMCalculator(
#         nir_path=nir_path, window_size=3, propery="ASM"
#     ).execute(output_path="./", title="GLCM output")
