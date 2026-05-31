import numpy as np

from fezrs.tools.image_enhancement.original_calculator import OriginalCalculator


def test_original_calculator_process(monkeypatch):
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
        self._output = None

    monkeypatch.setattr(OriginalCalculator, "__init__", mock_init)

    calculator = OriginalCalculator("dummy.tif")
    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == nir.shape
    assert result.dtype.kind == "f"


def test_original_calculator_execute(monkeypatch):
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
        self._output = None

    monkeypatch.setattr(OriginalCalculator, "__init__", mock_init)

    calculator = OriginalCalculator("dummy.tif")
    calculator.process()

    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


def test_original_calculator_histogram_export(tmp_path, monkeypatch):
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
        self._output = None

    monkeypatch.setattr(OriginalCalculator, "__init__", mock_init)
    monkeypatch.setattr(
        OriginalCalculator,
        "_add_watermark",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        OriginalCalculator,
        "_save_histogram_figure",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        OriginalCalculator,
        "_validate",
        lambda *args, **kwargs: None,
    )

    calculator = OriginalCalculator("dummy.tif")

    result = calculator.histogram_export(
        output_path=tmp_path,
        title="Original Histogram",
    )

    assert result is calculator
    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


# NOTE - These block code for integration test the OriginalCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = OriginalCalculator(nir_path=nir_path).execute("./", title="IE")
