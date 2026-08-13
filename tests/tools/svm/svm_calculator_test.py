import cv2
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


HEIGHT = 4
WIDTH = 7
N_BANDS = 6


def _unique_band(band_index):
    """Build a 4x7 band whose value encodes (band, row, col)."""
    rows, cols = np.indices((HEIGHT, WIDTH))
    return (band_index * 10000 + rows * 100 + cols).astype(float)


def _make_nonsquare_svm_calculator(class_number=2, sample_number=2):
    bands = [_unique_band(index) for index in range(N_BANDS)]
    fake_normalized_bands = {
        "red": bands[0] / bands[0].max(),
        "green": bands[1] / bands[1].max(),
        "blue": bands[2] / bands[2].max(),
    }
    fake_metadata = {"blue": {"height": HEIGHT, "width": WIDTH}}

    fake_files_handler = MagicMock()
    fake_files_handler.get_normalized_bands.return_value = fake_normalized_bands
    fake_files_handler.get_metadata_bands.return_value = fake_metadata
    fake_files_handler.get_images_collection.return_value = bands

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.normalized_bands = fake_normalized_bands
        self.metadata_shape = fake_metadata
        self.collection_bands = bands
        self.index_loop = 0
        self.is_finished_click_event = False
        self.class_number = kwargs.get("class_number", class_number)
        self.sample_number = kwargs.get("sample_number", sample_number)

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
            class_number=class_number,
            sample_number=sample_number,
        )

    return calculator, bands


def _run_process_with_clicks(calculator, clicks):
    callback_holder = {}
    click_index = {"value": 0}

    def fake_set_mouse_callback(_name, callback):
        callback_holder["callback"] = callback

    def fake_wait_key(_delay):
        callback = callback_holder["callback"]
        index = click_index["value"]
        if index < len(clicks):
            x, y = clicks[index]
            click_index["value"] += 1
            callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return 0
        return 27

    with patch("fezrs.tools.svm.svm_calculator.cv2.namedWindow"):
        with patch(
            "fezrs.tools.svm.svm_calculator.cv2.setMouseCallback",
            fake_set_mouse_callback,
        ):
            with patch("fezrs.tools.svm.svm_calculator.cv2.imshow"):
                with patch(
                    "fezrs.tools.svm.svm_calculator.cv2.waitKey",
                    fake_wait_key,
                ):
                    with patch(
                        "fezrs.tools.svm.svm_calculator.cv2.destroyAllWindows"
                    ):
                        calculator.process()


def test_nonsquare_training_samples_use_opencv_xy_as_numpy_yx():
    """
    OpenCV reports (x, y). Samples must be read from the raster as [y, x].

    Clicks include x >= height so the old [x][y] indexing cannot succeed.
    """
    calculator, bands = _make_nonsquare_svm_calculator(
        class_number=2,
        sample_number=2,
    )
    clicks = [
        (6, 0),
        (5, 1),
        (0, 3),
        (2, 2),
        (1, 1),
    ]

    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.ones(HEIGHT * WIDTH, dtype=int)

    with patch(
        "fezrs.tools.svm.svm_calculator.svm.SVC",
        return_value=mock_clf,
    ):
        _run_process_with_clicks(calculator, clicks)

    training_x, training_y = mock_clf.fit.call_args[0]
    expected_samples = np.array(
        [
            [band[y, x] for band in bands]
            for x, y in clicks[:4]
        ],
        dtype=object,
    )

    np.testing.assert_array_equal(training_x, expected_samples)
    np.testing.assert_array_equal(training_y, np.array([1, 1, 2, 2]))


def test_nonsquare_output_shape_and_pixel_order():
    calculator, bands = _make_nonsquare_svm_calculator(
        class_number=2,
        sample_number=2,
    )
    clicks = [
        (6, 0),
        (5, 1),
        (0, 3),
        (2, 2),
        (1, 1),
    ]

    mock_clf = MagicMock()

    def fake_predict(features):
        assert features.shape == (HEIGHT * WIDTH, N_BANDS)
        np.testing.assert_array_equal(
            features[0],
            [band[0, 0] for band in bands],
        )
        np.testing.assert_array_equal(
            features[1],
            [band[0, 1] for band in bands],
        )
        np.testing.assert_array_equal(
            features[WIDTH],
            [band[1, 0] for band in bands],
        )
        np.testing.assert_array_equal(
            features[-1],
            [band[HEIGHT - 1, WIDTH - 1] for band in bands],
        )
        return np.arange(len(features))

    mock_clf.predict.side_effect = fake_predict

    with patch(
        "fezrs.tools.svm.svm_calculator.svm.SVC",
        return_value=mock_clf,
    ):
        _run_process_with_clicks(calculator, clicks)

    assert calculator._output.shape == (HEIGHT, WIDTH)
    np.testing.assert_array_equal(
        calculator._output,
        np.arange(HEIGHT * WIDTH).reshape(HEIGHT, WIDTH),
    )
