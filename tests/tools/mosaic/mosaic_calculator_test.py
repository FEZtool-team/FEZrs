import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fezrs.tools.mosaic.mosaic_calculator import MosaicCalculator


@pytest.fixture
def mock_mosaic_calculator():
    fake_rasterio_tifs = [MagicMock(), MagicMock()]

    fake_rasterio_tifs[0].meta = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "crs": "EPSG:4326",
    }
    fake_rasterio_tifs[0].crs = "EPSG:4326"

    fake_rasterio_tifs[1].meta = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "crs": "EPSG:4326",
    }
    fake_rasterio_tifs[1].crs = "EPSG:4326"

    fake_files_handler = MagicMock()
    fake_files_handler.get_rasterio_tifs.return_value = fake_rasterio_tifs

    def fake_init(self, *args, **kwargs):
        self.files_handler = fake_files_handler
        self._output = None
        self.mosaic_rasterio_tifs = fake_rasterio_tifs

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.BaseTool.__init__",
        fake_init,
    ):
        calculator = MosaicCalculator(
            tif_paths=["dummy1.tif", "dummy2.tif"],
        )

    return calculator


def test_initialization(mock_mosaic_calculator):
    assert mock_mosaic_calculator.mosaic_rasterio_tifs is not None
    assert len(mock_mosaic_calculator.mosaic_rasterio_tifs) == 2
    assert mock_mosaic_calculator._output is None


def test_validate_method_exists(mock_mosaic_calculator):
    mock_mosaic_calculator._validate()


def test_process_creates_mosaic(mock_mosaic_calculator):
    mock_mosaic = np.random.rand(3, 200, 200)
    mock_transform = MagicMock()

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ) as mock_merge:
        mock_mosaic_calculator.process()

        mock_merge.assert_called_once_with(mock_mosaic_calculator.mosaic_rasterio_tifs)
        assert hasattr(mock_mosaic_calculator, "mosaic_meta")
        assert hasattr(mock_mosaic_calculator, "mosaic_mimg")
        assert np.array_equal(mock_mosaic_calculator.mosaic_mimg, mock_mosaic)


def test_process_updates_metadata_correctly(mock_mosaic_calculator):
    mock_mosaic = np.random.rand(3, 200, 200)
    mock_transform = MagicMock()

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ):
        mock_mosaic_calculator.process()

        assert mock_mosaic_calculator.mosaic_meta["driver"] == "GTiff"
        assert mock_mosaic_calculator.mosaic_meta["height"] == 200
        assert mock_mosaic_calculator.mosaic_meta["width"] == 200
        assert mock_mosaic_calculator.mosaic_meta["transform"] == mock_transform
        assert (
            mock_mosaic_calculator.mosaic_meta["crs"]
            == mock_mosaic_calculator.mosaic_rasterio_tifs[0].crs
        )


def test_export_file_creates_tif_and_png(mock_mosaic_calculator):
    mock_mosaic = np.random.rand(3, 100, 100)
    mock_transform = MagicMock()
    mock_mosaic_calculator.mosaic_mimg = mock_mosaic
    mock_mosaic_calculator.mosaic_meta = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
    }

    mock_src = MagicMock()
    mock_read_result = np.random.rand(100, 100)
    mock_src.read.return_value = mock_read_result

    mock_uuid = "1234567890abcdef"

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ):
        with patch("fezrs.tools.mosaic.mosaic_calculator.uuid4") as mock_uuid4:
            mock_uuid4.return_value.hex = mock_uuid
            mock_uuid4.side_effect = [
                MagicMock(hex=mock_uuid),
                MagicMock(hex=mock_uuid),
            ]

            with patch(
                "fezrs.tools.mosaic.mosaic_calculator.rio_open"
            ) as mock_rio_open:
                mock_dest = MagicMock()
                mock_context_dest = MagicMock()
                mock_context_dest.__enter__ = MagicMock(return_value=mock_dest)
                mock_context_dest.__exit__ = MagicMock(return_value=False)

                mock_context_src = MagicMock()
                mock_context_src.__enter__ = MagicMock(return_value=mock_src)
                mock_context_src.__exit__ = MagicMock(return_value=False)

                mock_rio_open.side_effect = [mock_context_dest, mock_context_src]

                with patch(
                    "fezrs.tools.mosaic.mosaic_calculator.plt.imsave"
                ) as mock_imsave:
                    with patch("pathlib.Path.mkdir") as mock_mkdir:
                        mock_mosaic_calculator.process()
                        result = mock_mosaic_calculator._export_file(
                            output_path="./output", colormap="viridis", dpi=150
                        )

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_dest.write.assert_called_once_with(mock_mosaic)
    mock_imsave.assert_called_once()
    assert mock_mosaic_calculator._output is not None
    assert ".tif" in mock_mosaic_calculator._output


def test_export_file_creates_output_directory(mock_mosaic_calculator):
    mock_mosaic = np.random.rand(3, 100, 100)
    mock_transform = MagicMock()
    mock_mosaic_calculator.mosaic_mimg = mock_mosaic
    mock_mosaic_calculator.mosaic_meta = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
    }

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ):
        with patch("fezrs.tools.mosaic.mosaic_calculator.rio_open") as mock_rio_open:
            mock_dest = MagicMock()
            mock_src = MagicMock()
            mock_src.read.return_value = np.random.rand(100, 100)

            mock_context_dest = MagicMock()
            mock_context_dest.__enter__ = MagicMock(return_value=mock_dest)
            mock_context_dest.__exit__ = MagicMock(return_value=False)

            mock_context_src = MagicMock()
            mock_context_src.__enter__ = MagicMock(return_value=mock_src)
            mock_context_src.__exit__ = MagicMock(return_value=False)

            mock_rio_open.side_effect = [mock_context_dest, mock_context_src]

            with patch("fezrs.tools.mosaic.mosaic_calculator.plt.imsave"):
                with patch("pathlib.Path.mkdir") as mock_mkdir:
                    with patch("fezrs.tools.mosaic.mosaic_calculator.uuid4"):
                        mock_mosaic_calculator.process()
                        mock_mosaic_calculator._export_file(output_path="./new_output")

                        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_export_file_uses_class_name_as_prefix(mock_mosaic_calculator):
    mock_mosaic = np.random.rand(3, 100, 100)
    mock_transform = MagicMock()
    mock_mosaic_calculator.mosaic_mimg = mock_mosaic
    mock_mosaic_calculator.mosaic_meta = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
    }

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ):
        with patch("fezrs.tools.mosaic.mosaic_calculator.rio_open") as mock_rio_open:
            mock_dest = MagicMock()
            mock_src = MagicMock()
            mock_src.read.return_value = np.random.rand(100, 100)

            mock_context_dest = MagicMock()
            mock_context_dest.__enter__ = MagicMock(return_value=mock_dest)
            mock_context_dest.__exit__ = MagicMock(return_value=False)

            mock_context_src = MagicMock()
            mock_context_src.__enter__ = MagicMock(return_value=mock_src)
            mock_context_src.__exit__ = MagicMock(return_value=False)

            mock_rio_open.side_effect = [mock_context_dest, mock_context_src]

            with patch("fezrs.tools.mosaic.mosaic_calculator.plt.imsave"):
                with patch("pathlib.Path.mkdir"):
                    with patch(
                        "fezrs.tools.mosaic.mosaic_calculator.uuid4"
                    ) as mock_uuid4:
                        mock_uuid4.return_value.hex = "test123"

                        mock_mosaic_calculator.process()
                        mock_mosaic_calculator._export_file(output_path="./output")

                        assert "Mosaic" in mock_mosaic_calculator._output


def test_execute_calls_base_execute(mock_mosaic_calculator):
    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.BaseTool.execute", return_value="executed"
    ) as mock_execute:
        result = mock_mosaic_calculator.execute(
            output_path="output/test",
            title="Mosaic Test",
            figsize=(15, 10),
            show_axis=True,
            colormap="viridis",
            show_colorbar=True,
            filename_prefix="MosaicTool",
            dpi=300,
            bbox_inches="tight",
            grid=True,
            nrows=2,
            ncols=2,
        )

    mock_execute.assert_called_once_with(
        "output/test",
        "Mosaic Test",
        (15, 10),
        True,
        "viridis",
        True,
        "MosaicTool",
        300,
        "tight",
        True,
        2,
        2,
    )
    assert result == "executed"


def test_execute_with_default_parameters(mock_mosaic_calculator):
    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.BaseTool.execute", return_value="executed"
    ) as mock_execute:
        result = mock_mosaic_calculator.execute("output.png")

    mock_execute.assert_called_once_with(
        "output.png",
        None,
        (10, 10),
        False,
        "gray",
        False,
        "Tool_output",
        100,
        "tight",
        False,
        None,
        None,
    )
    assert result == "executed"


def test_process_handles_multiple_tifs(mock_mosaic_calculator):
    assert len(mock_mosaic_calculator.mosaic_rasterio_tifs) == 2

    mock_mosaic = np.random.rand(3, 300, 300)
    mock_transform = MagicMock()

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ) as mock_merge:
        mock_mosaic_calculator.process()

        args, _ = mock_merge.call_args
        assert len(args[0]) == 2


def test_export_file_reads_first_band_for_png(mock_mosaic_calculator):
    mock_mosaic = np.random.rand(3, 100, 100)
    mock_transform = MagicMock()
    mock_mosaic_calculator.mosaic_mimg = mock_mosaic
    mock_mosaic_calculator.mosaic_meta = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
    }

    mock_src = MagicMock()
    mock_read_result = np.random.rand(100, 100)
    mock_src.read.return_value = mock_read_result

    with patch(
        "fezrs.tools.mosaic.mosaic_calculator.merge",
        return_value=(mock_mosaic, mock_transform),
    ):
        with patch("fezrs.tools.mosaic.mosaic_calculator.rio_open") as mock_rio_open:
            mock_dest = MagicMock()

            mock_context_dest = MagicMock()
            mock_context_dest.__enter__ = MagicMock(return_value=mock_dest)
            mock_context_dest.__exit__ = MagicMock(return_value=False)

            mock_context_src = MagicMock()
            mock_context_src.__enter__ = MagicMock(return_value=mock_src)
            mock_context_src.__exit__ = MagicMock(return_value=False)

            mock_rio_open.side_effect = [mock_context_dest, mock_context_src]

            with patch(
                "fezrs.tools.mosaic.mosaic_calculator.plt.imsave"
            ) as mock_imsave:
                with patch("pathlib.Path.mkdir"):
                    with patch("fezrs.tools.mosaic.mosaic_calculator.uuid4"):
                        mock_mosaic_calculator.process()
                        mock_mosaic_calculator._export_file(output_path="./output")

                        mock_src.read.assert_called_once_with(1)
                        call_args = mock_imsave.call_args
                        assert np.array_equal(call_args[0][1], mock_read_result)
