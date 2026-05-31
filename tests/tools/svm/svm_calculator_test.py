import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.svm.svm_calculator import SVMCalculator


@pytest.fixture
def mock_svm_calculator():
    fake_normalized_bands = {
        "red": np.random.rand(100, 100),
        "green": np.random.rand(100, 100),
        "blue": np.random.rand(100, 100),
    }

    fake_metadata = {"blue": {"height": 100, "width": 100}}

    fake_collection_bands = [
        np.random.rand(100, 100),
        np.random.rand(100, 100),
        np.random.rand(100, 100),
        np.random.rand(100, 100),
        np.random.rand(100, 100),
        np.random.rand(100, 100),
    ]

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands
    fake_files_handler.get_metadata_bands.return_value = fake_metadata
    fake_files_handler.get_images_collection.return_value = fake_collection_bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands
        self.metadata_shape = fake_metadata
        self.collection_bands = fake_collection_bands
        self.index_loop = 0
        self.is_finished_click_event = False
        self.class_number = kwargs.get("class_number", 4)
        self.sample_number = kwargs.get("sample_number", 10)

    with patch(
        "fezrs.tools.svm.svm_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = SVMCalculator(
            red_path="dummy_red.tif",
            green_path="dummy_green.tif",
            blue_path="dummy_blue.tif",
            nir_path="dummy_nir.tif",
            swir1_path="dummy_swir1.tif",
            swir2_path="dummy_swir2.tif",
            class_number=4,
            sample_number=10,
        )

    return calculator


def test_initialization(mock_svm_calculator):
    assert mock_svm_calculator.normalized_bands is not None
    assert mock_svm_calculator.metadata_shape is not None
    assert mock_svm_calculator.collection_bands is not None
    assert mock_svm_calculator.index_loop == 0
    assert not mock_svm_calculator.is_finished_click_event
    assert mock_svm_calculator.class_number == 4
    assert mock_svm_calculator.sample_number == 10
    assert mock_svm_calculator._output is None


def test_validate_with_valid_parameters(mock_svm_calculator):
    mock_svm_calculator._validate()


def test_validate_raises_error_for_invalid_class_number_type(mock_svm_calculator):
    mock_svm_calculator.class_number = "invalid"
    with pytest.raises(ValueError, match="class_number must be an int."):
        mock_svm_calculator._validate()


def test_validate_raises_error_for_class_number_less_than_2(mock_svm_calculator):
    mock_svm_calculator.class_number = 1
    with pytest.raises(ValueError, match="class_number must be at least 2."):
        mock_svm_calculator._validate()


def test_validate_raises_error_for_invalid_sample_number_type(mock_svm_calculator):
    mock_svm_calculator.sample_number = "invalid"
    with pytest.raises(ValueError, match="sample_number must be an int."):
        mock_svm_calculator._validate()


def test_validate_raises_error_for_sample_number_less_than_1(mock_svm_calculator):
    mock_svm_calculator.sample_number = 0
    with pytest.raises(ValueError, match="sample_number must be at least 1."):
        mock_svm_calculator._validate()


def test_validate_raises_error_when_samples_exceed_pixels(mock_svm_calculator):
    mock_svm_calculator.metadata_shape = {"blue": {"height": 10, "width": 10}}
    mock_svm_calculator.class_number = 10
    mock_svm_calculator.sample_number = 20

    with pytest.raises(
        ValueError, match="Requested 200 samples, but the image only has 100 pixels."
    ):
        mock_svm_calculator._validate()


def test_process_creates_rgb_stack(mock_svm_calculator):
    with patch("fezrs.tools.svm.svm_calculator.io.concatenate_images") as mock_concat:
        mock_concat.return_value = MagicMock()
        with patch("fezrs.tools.svm.svm_calculator.cv2.namedWindow"):
            with patch("fezrs.tools.svm.svm_calculator.cv2.setMouseCallback"):
                with patch("fezrs.tools.svm.svm_calculator.cv2.imshow"):
                    with patch(
                        "fezrs.tools.svm.svm_calculator.cv2.waitKey",
                        return_value=27,
                    ):
                        with patch(
                            "fezrs.tools.svm.svm_calculator.cv2.destroyAllWindows"
                        ):
                            mock_svm_calculator.process()

                            assert (
                                mock_svm_calculator.normalized_bands["red"] is not None
                            )
                            assert (
                                mock_svm_calculator.normalized_bands["green"]
                                is not None
                            )
                            assert (
                                mock_svm_calculator.normalized_bands["blue"] is not None
                            )


def test_execute_calls_base_execute(mock_svm_calculator):
    with patch(
        "fezrs.tools.svm.svm_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_svm_calculator.execute(
            output_path="output/test.png",
            title="SVM Test",
            figsize=[15, 10],
            show_axis=True,
            colormap="viridis",
            show_colorbar=True,
            filename_prefix="SVMTool",
            dpi=300,
            bbox_inches="tight",
            grid=False,
            nrows=2,
            ncols=2,
        )

    mock_execute.assert_called_once_with(
        "output/test.png",
        "SVM Test",
        [15, 10],
        True,
        "viridis",
        True,
        "SVMTool",
        300,
        "tight",
        False,
        2,
        2,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_svm_calculator):
    with patch(
        "fezrs.tools.svm.svm_calculator.BaseTool.execute",
        return_value="executed",
    ) as mock_execute:
        result = mock_svm_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        "SVM",
        [10, 10],
        True,
        None,
        False,
        "Tool_output",
        500,
        "tight",
        False,
        None,
        None,
    )
    assert result == "executed"


def test_export_file_calls_base_export_file(mock_svm_calculator):
    with patch(
        "fezrs.tools.svm.svm_calculator.BaseTool._export_file",
        return_value="exported",
    ) as mock_export:
        result = mock_svm_calculator._export_file(
            output_path="output/test.png",
            title="SVM Export",
            figsize=[12, 8],
            show_axis=False,
            colormap="gray",
            show_colorbar=True,
            filename_prefix="SVMExport",
            dpi=200,
            bbox_inches="tight",
            grid=True,
            nrows=1,
            ncols=1,
        )

    mock_export.assert_called_once_with(
        "output/test.png",
        "SVM Export",
        [12, 8],
        False,
        "gray",
        True,
        "SVMExport",
        200,
        "tight",
        True,
        1,
        1,
    )
    assert result == "exported"
