import numpy as np

from fezrs.tools.image_enhancement.gamma_calculator import GammaCalculator


def test_gamma_calculator_process(monkeypatch):
    nir = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )

    metadata = {
        "nir": {
            "image_skimage": nir,
            "width": 2,
            "height": 2,
        }
    }

    def mock_init(self, nir_path):
        self.metadata_bands = metadata
        self.gamma = 0.2
        self.gain = 1
        self._output = None

    monkeypatch.setattr(GammaCalculator, "__init__", mock_init)

    calculator = GammaCalculator("dummy.tif")
    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == nir.shape
    assert result.dtype.kind == "f"


def test_gamma_calculator_execute(tmp_path, monkeypatch):
    nir = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )

    metadata = {
        "nir": {
            "image_skimage": nir,
            "width": 2,
            "height": 2,
        }
    }

    def mock_init(self, nir_path, gamma=0.2, gain=1):
        self.metadata_bands = metadata
        self.gamma = gamma
        self.gain = gain
        self._output = None

    monkeypatch.setattr(GammaCalculator, "__init__", mock_init)

    calculator = GammaCalculator("dummy.tif")
    calculator.process()

    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


def test_gamma_calculator_histogram_export(tmp_path, monkeypatch):
    nir = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )

    metadata = {
        "nir": {
            "image_skimage": nir,
            "width": 2,
            "height": 2,
        }
    }

    def mock_init(self, nir_path, gamma=0.2, gain=1):
        self.metadata_bands = metadata
        self.gamma = gamma
        self.gain = gain
        self._output = None

    monkeypatch.setattr(GammaCalculator, "__init__", mock_init)
    monkeypatch.setattr(GammaCalculator, "_add_watermark", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        GammaCalculator,
        "_save_histogram_figure",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(GammaCalculator, "_validate", lambda *args, **kwargs: None)

    calculator = GammaCalculator("dummy.tif")

    result = calculator.histogram_export(
        output_path=tmp_path,
        title="Gamma Histogram",
    )

    assert result is calculator
    assert calculator._output is not None


# NOTE - These block code for integration test the GammaCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = GammaCalculator(nir_path=nir_path, gamma=0.2, gain=1).histogram_export(
#         "./", title="Gamma IE"
#     )
