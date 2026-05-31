import numpy as np

from fezrs.tools.image_enhancement.original_rgb_calculator import (
    OriginalRGBCalculator,
)


def test_original_rgb_calculator_process(monkeypatch):
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    green = np.array([[0.2, 0.3], [0.4, 0.5]])
    blue = np.array([[0.3, 0.4], [0.5, 0.6]])

    normalized_bands = {
        "red": red,
        "green": green,
        "blue": blue,
    }

    def mock_init(self, red_path, green_path, blue_path):
        self.normalized_bands = normalized_bands
        self._output = None

    monkeypatch.setattr(
        OriginalRGBCalculator,
        "__init__",
        mock_init,
    )

    calculator = OriginalRGBCalculator(
        "red.tif",
        "green.tif",
        "blue.tif",
    )

    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 3)

    np.testing.assert_array_equal(result[:, :, 0], red)
    np.testing.assert_array_equal(result[:, :, 1], green)
    np.testing.assert_array_equal(result[:, :, 2], blue)


def test_original_rgb_calculator_execute(monkeypatch):
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    green = np.array([[0.2, 0.3], [0.4, 0.5]])
    blue = np.array([[0.3, 0.4], [0.5, 0.6]])

    normalized_bands = {
        "red": red,
        "green": green,
        "blue": blue,
    }

    def mock_init(self, red_path, green_path, blue_path):
        self.normalized_bands = normalized_bands
        self._output = None

    monkeypatch.setattr(
        OriginalRGBCalculator,
        "__init__",
        mock_init,
    )

    calculator = OriginalRGBCalculator(
        "red.tif",
        "green.tif",
        "blue.tif",
    )

    calculator.process()

    assert calculator._output is not None
    assert calculator._output.shape == (2, 2, 3)


def test_original_rgb_calculator_histogram_export(tmp_path, monkeypatch):
    red = np.array([[0.1, 0.2], [0.3, 0.4]])
    green = np.array([[0.2, 0.3], [0.4, 0.5]])
    blue = np.array([[0.3, 0.4], [0.5, 0.6]])

    normalized_bands = {
        "red": red,
        "green": green,
        "blue": blue,
    }

    def mock_init(self, red_path, green_path, blue_path):
        self.normalized_bands = normalized_bands
        self._output = None

    monkeypatch.setattr(
        OriginalRGBCalculator,
        "__init__",
        mock_init,
    )

    monkeypatch.setattr(
        OriginalRGBCalculator,
        "_add_watermark",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        OriginalRGBCalculator,
        "_save_histogram_figure",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        OriginalRGBCalculator,
        "_validate",
        lambda *args, **kwargs: None,
    )

    calculator = OriginalRGBCalculator(
        "red.tif",
        "green.tif",
        "blue.tif",
    )

    result = calculator.histogram_export(
        output_path=tmp_path,
        title="RGB Histogram",
    )

    assert result is calculator
    assert calculator._output is not None
    assert calculator._output.shape == (2, 2, 3)


# NOTE - These block code for integration test the OriginalRGBCalculator
# if __name__ == "__main__":
#     red_path = Path.cwd() / "data/Red.tif"
#     green_path = Path.cwd() / "data/Green.tif"
#     blue_path = Path.cwd() / "data/Blue.tif"
#
#     calculator = OriginalRGBCalculator(
#         red_path=red_path,
#         green_path=green_path,
#         blue_path=blue_path,
#     ).execute("./", title="RGB IE")
