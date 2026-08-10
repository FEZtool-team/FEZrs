import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sklearn.decomposition import PCA

from fezrs.tools.pca.pca_calculator import PCACalculator


@pytest.fixture
def mock_pca_calculator():
    """
    Create a PCACalculator using deterministic, non-square test data.

    The raster shape is intentionally 4x7 so that width/height
    transposition bugs cannot be hidden by a square raster.
    """
    height = 4
    width = 7

    fake_metadata = {
        "red": {"width": width, "height": height},
        "green": {"width": width, "height": height},
        "blue": {"width": width, "height": height},
        "nir": {"width": width, "height": height},
        "swir1": {"width": width, "height": height},
        "swir2": {"width": width, "height": height},
    }

    # Deterministic and asymmetric data.
    # Every band has different values so that transposition/order
    # mistakes are easy to detect.
    red = np.arange(1, 29, dtype=float).reshape(height, width)
    green = np.arange(101, 129, dtype=float).reshape(height, width)
    blue = np.arange(201, 229, dtype=float).reshape(height, width)
    nir = np.arange(301, 329, dtype=float).reshape(height, width)
    swir1 = np.arange(401, 429, dtype=float).reshape(height, width)
    swir2 = np.arange(501, 529, dtype=float).reshape(height, width)

    fake_images_collection = [
        red,
        green,
        blue,
        nir,
        swir1,
        swir2,
    ]

    fake_files_handler = MagicMock()
    fake_files_handler.get_metadata_bands.return_value = fake_metadata
    fake_files_handler.get_images_collection.return_value = (
        fake_images_collection
    )

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.metadata_bands = fake_metadata
        self.image_shape = (height, width)
        self.selectBand = kwargs.get("selectBand")

        self._logo_watermark = None

        self.bindTheBandsToNumber = {
            "red": 0,
            "nir": 1,
            "blue": 2,
            "swir1": 3,
            "swir2": 4,
            "green": 5,
        }

    with patch(
        "fezrs.tools.pca.pca_calculator.BaseTool.__init__",
        fake_init,
    ):
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
    assert mock_pca_calculator.image_shape == (4, 7)


def test_process_output_shape(mock_pca_calculator):
    """
    PCA output must contain one 4x7 raster for each principal component.
    """
    result = mock_pca_calculator.process()

    assert result is not None
    assert len(result) == 6

    for component in result:
        assert component.shape == (4, 7)


def test_process_numerical_values(mock_pca_calculator):
    """
    Compare PCACalculator's result against an independently-created
    sklearn PCA reference.

    Each pixel is a sample and each spectral band is a feature:

        (number_of_pixels, number_of_bands)
        (28, 6)
    """
    result = mock_pca_calculator.process()

    images = mock_pca_calculator.files_handler.get_images_collection()

    expected_input = np.stack(
        images,
        axis=-1,
    ).reshape(-1, 6)

    assert expected_input.shape == (28, 6)

    reference_pca = PCA(n_components=6)
    expected = reference_pca.fit_transform(expected_input)

    expected_components = [
        expected[:, index].reshape(4, 7)
        for index in range(6)
    ]

    for actual, expected_component in zip(
        result,
        expected_components,
    ):
        # PCA component signs can theoretically be flipped.
        # Since both calculations use the same sklearn implementation,
        # they should normally have identical signs, but allowing a
        # sign flip makes this test robust to equivalent PCA solutions.
        if not np.allclose(actual, expected_component):
            np.testing.assert_allclose(
                actual,
                -expected_component,
            )


def test_histogram_export_requires_select_band(mock_pca_calculator):
    mock_pca_calculator.selectBand = None

    with pytest.raises(
        ValueError,
        match="You cannot use histogram_export\\(\\) without "
        "passing selectBand\\.",
    ):
        mock_pca_calculator.histogram_export(
            output_path="./output"
        )


def test_histogram_export_success(mock_pca_calculator, tmp_path):
    mock_pca_calculator.process()

    output_path = tmp_path / "pca"

    with patch.object(
        mock_pca_calculator,
        "_save_histogram_figure",
    ) as mock_save_histogram:

        with patch.object(
            mock_pca_calculator,
            "_add_watermark",
        ):

            result = mock_pca_calculator.histogram_export(
                output_path=output_path,
                title="Test",
            )

    assert result is mock_pca_calculator
    mock_save_histogram.assert_called_once()


def test_export_file(mock_pca_calculator, tmp_path):
    mock_pca_calculator.process()

    output_path = tmp_path / "pca"

    with patch(
        "fezrs.tools.pca.pca_calculator.plt.subplots"
    ) as mock_subplots:

        mock_fig = MagicMock()

        mock_axes = np.empty((6, 2), dtype=object)

        for row in range(6):
            for column in range(2):
                mock_axes[row, column] = MagicMock()

        mock_subplots.return_value = (
            mock_fig,
            mock_axes,
        )

        mock_pca_calculator._export_file(
            output_path=output_path,
        )

        mock_fig.savefig.assert_called_once()


def test_execute(mock_pca_calculator):
    """
    Verify that execute() delegates to BaseTool.execute()
    instead of mocking execute() itself.
    """
    with patch(
        "fezrs.tools.pca.pca_calculator.BaseTool.execute",
        return_value=mock_pca_calculator,
    ) as mock_execute:

        result = mock_pca_calculator.execute(
            output_path="./output",
        )

    assert result is mock_pca_calculator
    mock_execute.assert_called_once()
