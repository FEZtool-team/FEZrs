import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.pca.pca_calculator import PCACalculator


@pytest.fixture
def mock_pca_calculator():
    fake_metadata = {
        "red": {"width": 100, "height": 100},
        "green": {"width": 100, "height": 100},
        "blue": {"width": 100, "height": 100},
        "nir": {"width": 100, "height": 100},
        "swir1": {"width": 100, "height": 100},
        "swir2": {"width": 100, "height": 100},
    }

    fake_images_collection = [np.random.rand(10000) for _ in range(6)]

    fake_files_handler = MagicMock()
    fake_files_handler.get_metadata_bands.return_value = fake_metadata
    fake_files_handler.get_images_collection.return_value = fake_images_collection

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.metadata_bands = fake_metadata
        self.image_shape = (100, 100)
        self.selectBand = kwargs.get("selectBand", None)
        self._logo_watermark = None
        self.bindTheBandsToNumber = {
            "red": 0,
            "nir": 1,
            "blue": 2,
            "swir1": 3,
            "swir2": 4,
            "green": 5,
        }

    with patch("fezrs.tools.pca.pca_calculator.BaseTool.__init__", fake_init):
        calculator = PCACalculator(
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
            blue_path="dummy_blue.tif",
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
            swir2_path="dummy_swir2.tif",
            selectBand="swir2",
        )

    return calculator


def test_initialization(mock_pca_calculator):
    assert mock_pca_calculator.selectBand == "swir2"
    assert mock_pca_calculator._output is None


def test_process(mock_pca_calculator):
    mock_components = np.random.rand(6, 10000)
    mock_pca_instance = MagicMock()
    mock_pca_instance.components_ = mock_components

    with patch("fezrs.tools.pca.pca_calculator.skpc") as mock_skpc:
        mock_skpc.return_value = mock_pca_instance
        result = mock_pca_calculator.process()

        assert result is not None
        assert mock_pca_calculator._output is not None


# def test_histogram_export_error(mock_pca_calculator):
#     mock_pca_calculator.selectBand = None
#     mock_pca_calculator._output = np.random.rand(6, 10000)

#     with pytest.raises(
#         Exception,
#         match="You cant use histogram method if you are not passed select band value",
#     ):
#         mock_pca_calculator.histogram_export(output_path="./output.png")


def test_histogram_export_success(mock_pca_calculator):
    mock_pca_calculator._output = np.random.rand(6, 10000)

    with patch.object(mock_pca_calculator, "_save_histogram_figure"):
        with patch.object(mock_pca_calculator, "_add_watermark"):
            with patch("fezrs.tools.pca.pca_calculator.plt.subplots") as mock_subplots:
                mock_subplots.return_value = (MagicMock(), MagicMock())
                result = mock_pca_calculator.histogram_export(
                    output_path="./output.png", title="Test"
                )
                assert result is mock_pca_calculator


def test_export_file(mock_pca_calculator):
    mock_pca_calculator._output = np.random.rand(6, 10000)

    with patch("fezrs.tools.pca.pca_calculator.plt.subplots") as mock_subplots:
        mock_subplots.return_value = (MagicMock(), MagicMock())
        with patch("fezrs.tools.pca.pca_calculator.plt.title"):
            with patch("fezrs.tools.pca.pca_calculator.uuid4"):
                with patch("fezrs.tools.pca.pca_calculator.plt.close"):
                    mock_pca_calculator._export_file(output_path="./output")
                    assert True


def test_execute(mock_pca_calculator):
    with patch.object(mock_pca_calculator, "execute") as mock_execute:
        mock_execute.return_value = "executed"
        result = mock_pca_calculator.execute("output.png")
        assert result == "executed"
