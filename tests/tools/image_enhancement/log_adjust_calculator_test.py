import numpy as np

from fezrs.tools.image_enhancement.log_adjust_calculator import LogAdjustCalculator


def test_log_adjust_calculator_process(monkeypatch):
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

    def mock_init(self, nir_path, gain=1, inverse=False):
        self.metadata_bands = metadata
        self.gain = gain
        self.inverse = inverse
        self._output = None

    monkeypatch.setattr(LogAdjustCalculator, "__init__", mock_init)

    calculator = LogAdjustCalculator("dummy.tif")
    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == nir.shape
    assert result.dtype.kind == "f"


def test_log_adjust_calculator_execute(monkeypatch):
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

    def mock_init(self, nir_path, gain=1, inverse=False):
        self.metadata_bands = metadata
        self.gain = gain
        self.inverse = inverse
        self._output = None

    monkeypatch.setattr(LogAdjustCalculator, "__init__", mock_init)

    calculator = LogAdjustCalculator("dummy.tif")
    calculator.process()

    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


def test_log_adjust_calculator_histogram_export(tmp_path, monkeypatch):
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

    def mock_init(self, nir_path, gain=1, inverse=False):
        self.metadata_bands = metadata
        self.gain = gain
        self.inverse = inverse
        self._output = None

    monkeypatch.setattr(LogAdjustCalculator, "__init__", mock_init)
    monkeypatch.setattr(
        LogAdjustCalculator,
        "_add_watermark",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        LogAdjustCalculator,
        "_save_histogram_figure",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        LogAdjustCalculator,
        "_validate",
        lambda *args, **kwargs: None,
    )

    calculator = LogAdjustCalculator("dummy.tif")

    result = calculator.histogram_export(
        output_path=tmp_path,
        title="Log Adjust Histogram",
    )

    assert result is calculator
    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


# NOTE - These block code for integration test the LogAdjustCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"

#     calculator = (
#         LogAdjustCalculator(nir_path=nir_path, inverse=False, gain=1)
#         .histogram_export("./", title="LogAdjust IE")
#         .execute("./")
#     )
