import numpy as np

from fezrs.tools.image_enhancement.sigmoid_adjust_calculator import (
    SigmoidAdjustCalculator,
)


def test_sigmoid_adjust_calculator_process(monkeypatch):
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

    def mock_init(self, nir_path, gain=1, cutoff=0.5, inverse=False):
        self.metadata_bands = metadata
        self.gain = gain
        self.cutoff = cutoff
        self.inverse = inverse
        self._output = None

    monkeypatch.setattr(
        SigmoidAdjustCalculator,
        "__init__",
        mock_init,
    )

    calculator = SigmoidAdjustCalculator(
        "dummy.tif",
        gain=10,
        cutoff=0.5,
        inverse=False,
    )

    result = calculator.process()

    assert isinstance(result, np.ndarray)
    assert result.shape == nir.shape
    assert result.dtype.kind == "f"


def test_sigmoid_adjust_calculator_execute(monkeypatch):
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

    def mock_init(self, nir_path, gain=1, cutoff=0.5, inverse=False):
        self.metadata_bands = metadata
        self.gain = gain
        self.cutoff = cutoff
        self.inverse = inverse
        self._output = None

    monkeypatch.setattr(
        SigmoidAdjustCalculator,
        "__init__",
        mock_init,
    )

    calculator = SigmoidAdjustCalculator(
        "dummy.tif",
        gain=10,
        cutoff=0.5,
        inverse=False,
    )

    calculator.process()

    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


def test_sigmoid_adjust_calculator_histogram_export(
    tmp_path,
    monkeypatch,
):
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

    def mock_init(self, nir_path, gain=1, cutoff=0.5, inverse=False):
        self.metadata_bands = metadata
        self.gain = gain
        self.cutoff = cutoff
        self.inverse = inverse
        self._output = None

    monkeypatch.setattr(
        SigmoidAdjustCalculator,
        "__init__",
        mock_init,
    )

    monkeypatch.setattr(
        SigmoidAdjustCalculator,
        "_add_watermark",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        SigmoidAdjustCalculator,
        "_save_histogram_figure",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        SigmoidAdjustCalculator,
        "_validate",
        lambda *args, **kwargs: None,
    )

    calculator = SigmoidAdjustCalculator(
        "dummy.tif",
        gain=10,
        cutoff=0.5,
        inverse=False,
    )

    result = calculator.histogram_export(
        output_path=tmp_path,
        title="Sigmoid Histogram",
    )

    assert result is calculator
    assert calculator._output is not None
    assert calculator._output.shape == nir.shape


# NOTE - These block code for integration test the SigmoidAdjustCalculator
# if __name__ == "__main__":
#     nir_path = Path.cwd() / "data/NIR.tif"
#
#     calculator = SigmoidAdjustCalculator(
#         nir_path=nir_path,
#         inverse=False,
#         gain=10,
#         cutoff=0.5,
#     ).execute("./", title="Sigmoid Adjust IE")
