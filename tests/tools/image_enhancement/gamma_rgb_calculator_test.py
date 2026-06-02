import numpy as np

from fezrs.tools.image_enhancement.gamma_rgb_calculator import GammaRGBCalculator


def test_gamma_rgb_calculator_process(monkeypatch):
    red = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    green = np.array([[0.1, 0.3], [0.5, 0.7]], dtype=np.float32)
    blue = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)

    normalized_bands = {
        "red": red,
        "green": green,
        "blue": blue,
    }

    def mock_init(self, red_path, green_path, blue_path):
        self.normalized_bands = normalized_bands
        self._output = None

    monkeypatch.setattr(GammaRGBCalculator, "__init__", mock_init)

    calculator = GammaRGBCalculator("red.tif", "green.tif", "blue.tif")
    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 3)
    assert np.array_equal(result, calculator._output)


def test_gamma_rgb_calculator_execute(monkeypatch):
    red = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    green = np.array([[0.1, 0.3], [0.5, 0.7]], dtype=np.float32)
    blue = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)

    normalized_bands = {
        "red": red,
        "green": green,
        "blue": blue,
    }

    def mock_init(self, red_path, green_path, blue_path):
        self.normalized_bands = normalized_bands
        self._output = None

    monkeypatch.setattr(GammaRGBCalculator, "__init__", mock_init)

    calculator = GammaRGBCalculator("red.tif", "green.tif", "blue.tif")
    calculator.process()

    assert calculator._output is not None
    assert calculator._output.shape == (2, 2, 3)


def test_gamma_rgb_calculator_histogram_export(tmp_path, monkeypatch):
    red = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    green = np.array([[0.1, 0.3], [0.5, 0.7]], dtype=np.float32)
    blue = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)

    normalized_bands = {
        "red": red,
        "green": green,
        "blue": blue,
    }

    def mock_init(self, red_path, green_path, blue_path):
        self.normalized_bands = normalized_bands
        self._output = None

    monkeypatch.setattr(GammaRGBCalculator, "__init__", mock_init)
    monkeypatch.setattr(
        GammaRGBCalculator,
        "_add_watermark",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        GammaRGBCalculator,
        "_save_histogram_figure",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        GammaRGBCalculator,
        "_validate",
        lambda *args, **kwargs: None,
    )

    calculator = GammaRGBCalculator("red.tif", "green.tif", "blue.tif")

    result = calculator.histogram_export(
        output_path=tmp_path,
        title="Gamma RGB Histogram",
    )

    assert result is calculator
    assert calculator._output is not None
    assert calculator._output.shape == (2, 2, 3)


# NOTE - These block code for integration test the GammaRGBCalculator
# if __name__ == "__main__":
#     red_path = Path.cwd() / "data/Red.tif"
#     green_path = Path.cwd() / "data/Green.tif"
#     blue_path = Path.cwd() / "data/Blue.tif"

#     calculator = GammaRGBCalculator(
#         red_path=red_path, green_path=green_path, blue_path=blue_path
#     ).histogram_export("./", title="Gamma-RGB IE")
