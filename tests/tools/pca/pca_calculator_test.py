import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.pca.pca_calculator import PCACalculator


HEIGHT = 4
WIDTH = 7
N_BANDS = 6


def _nonsquare_bands():
    """
    Deterministic 4x7 bands with linearly independent spatial patterns.

    Affine copies of a single ramp (for example arange + offset) collapse
    to rank 1 after centering and cannot detect a sample/feature swap.
    """
    rows, cols = np.indices((HEIGHT, WIDTH), dtype=float)
    return [
        rows,
        cols,
        rows * cols,
        rows**2,
        cols**2,
        rows**2 * cols,
    ]


def _numpy_pca_scores(images, height, width):
    """
    Independent pixel-as-sample PCA using NumPy SVD, not sklearn.

    X has shape (height * width, n_bands). Scores are reshaped to
    (n_bands, height, width). Component signs may differ from sklearn.
    """
    n_bands = len(images)
    samples = np.stack([np.asarray(image) for image in images], axis=-1)
    samples = samples.reshape(-1, n_bands).astype(np.float64)
    centered = samples - samples.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt.T
    return scores.T.reshape(n_bands, height, width)


def _align_component_sign(actual, expected):
    if np.dot(actual.ravel(), expected.ravel()) < 0:
        return -expected
    return expected


@pytest.fixture
def mock_pca_calculator():
    """
    Create a PCACalculator using deterministic, non-square test data.

    The raster shape is intentionally 4x7 so that width/height
    transposition bugs cannot be hidden by a square raster.
    """
    height = HEIGHT
    width = WIDTH

    fake_metadata = {
        "red": {"width": width, "height": height},
        "green": {"width": width, "height": height},
        "blue": {"width": width, "height": height},
        "nir": {"width": width, "height": height},
        "swir1": {"width": width, "height": height},
        "swir2": {"width": width, "height": height},
    }

    fake_images_collection = _nonsquare_bands()

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
        self.component = kwargs.get("component")

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
    assert result.shape == (N_BANDS, HEIGHT, WIDTH)

    for component in result:
        assert component.shape == (HEIGHT, WIDTH)


def test_process_numerical_values(mock_pca_calculator):
    """
    Compare PCACalculator against an independent NumPy SVD reference.

    Each pixel is a sample and each spectral band is a feature:

        (number_of_pixels, number_of_bands)
        (28, 6)
    """
    result = mock_pca_calculator.process()

    images = mock_pca_calculator.files_handler.get_images_collection()
    expected_input = np.stack(images, axis=-1).reshape(-1, N_BANDS)
    assert expected_input.shape == (HEIGHT * WIDTH, N_BANDS)

    expected_components = _numpy_pca_scores(images, HEIGHT, WIDTH)

    for actual, expected_component in zip(result, expected_components):
        aligned = _align_component_sign(actual, expected_component)
        np.testing.assert_allclose(actual, aligned, atol=1e-8, rtol=1e-6)


def test_process_pixels_are_samples_not_bands(mock_pca_calculator):
    """
    With 28 independent 6-band pixels the covariance has rank 6, so every
    explained-variance entry is nonzero. Treating the 6 bands as samples
    would center a 6-row matrix and force the last component to ~0.
    """
    mock_pca_calculator.process()

    explained = mock_pca_calculator._pca.explained_variance_
    assert explained.shape == (N_BANDS,)
    assert np.all(explained > 1e-8)

    images = mock_pca_calculator.files_handler.get_images_collection()
    band_as_sample = np.stack(images, axis=0).reshape(N_BANDS, -1)
    centered_bands = band_as_sample - band_as_sample.mean(axis=0, keepdims=True)
    # 6 samples, 28 features: rank is at most 5 after centering
    assert np.linalg.matrix_rank(centered_bands) <= N_BANDS - 1


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


def test_histogram_export_runs_process_when_output_is_none(
    mock_pca_calculator, tmp_path
):
    assert mock_pca_calculator._output is None

    with patch.object(mock_pca_calculator, "_save_histogram_figure"):
        with patch.object(mock_pca_calculator, "_add_watermark"):
            mock_pca_calculator.histogram_export(output_path=tmp_path / "pca")

    assert mock_pca_calculator._output is not None
    assert mock_pca_calculator._output.shape == (N_BANDS, HEIGHT, WIDTH)


def test_histogram_export_accepts_component(mock_pca_calculator, tmp_path):
    mock_pca_calculator.selectBand = None
    mock_pca_calculator.component = 1

    with patch.object(
        mock_pca_calculator, "_save_histogram_figure"
    ) as mock_save:
        with patch.object(mock_pca_calculator, "_add_watermark"):
            mock_pca_calculator.histogram_export(output_path=tmp_path / "pca")

    mock_save.assert_called_once()


def test_explained_variance_ratio_requires_process(mock_pca_calculator):
    with pytest.raises(ValueError, match="PCA has not been computed"):
        _ = mock_pca_calculator.explained_variance_ratio


def test_explained_variance_ratio_after_process(mock_pca_calculator):
    mock_pca_calculator.process()
    ratio = mock_pca_calculator.explained_variance_ratio
    assert ratio.shape == (N_BANDS,)
    np.testing.assert_allclose(ratio.sum(), 1.0, atol=1e-6)
