from pathlib import Path

import numpy as np
import pytest

from fezrs.base import BaseTool


class DummyTool(BaseTool):
    def __init__(self):
        self._output = None
        self._validate_called = False
        self._process_called = False

    def _validate(self):
        self._validate_called = True

    def process(self):
        self._process_called = True
        self._output = np.array([[1, 2], [3, 4]])
        return self._output

    def _export_file(self, *args, **kwargs):
        return "dummy.png"


class DummyExportTool(BaseTool):
    def __init__(self):
        self._output = None
        self._BaseTool__tool_name = "Dummy"

    def _validate(self):
        pass

    def process(self):
        pass


def test_validate_raises_not_implemented():
    tool = BaseTool.__new__(BaseTool)

    with pytest.raises(NotImplementedError):
        tool._validate()


def test_process_raises_not_implemented():
    tool = BaseTool.__new__(BaseTool)

    with pytest.raises(NotImplementedError):
        tool.process()


def test_export_file_raises_when_output_is_none(tmp_path):
    tool = DummyExportTool()

    with pytest.raises(ValueError, match="Data not computed"):
        tool._export_file(tmp_path)


def test_export_file_creates_png(tmp_path):
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])

    filename = tool._export_file(tmp_path)

    assert filename.endswith(".png")


def test_export_file_creates_output_directory(tmp_path):
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])

    output_dir = tmp_path / "exports"

    tool._export_file(output_dir)

    assert output_dir.exists()


def test_execute_calls_validate_and_process():
    tool = DummyTool()

    tool.execute(".")

    assert tool._validate_called is True
    assert tool._process_called is True


def test_execute_returns_self():
    tool = DummyTool()

    result = tool.execute(".")

    assert result is tool


def test_export_file_honors_filename_prefix(tmp_path):
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])

    filename = tool._export_file(tmp_path, filename_prefix="mine")

    assert Path(filename).name.startswith("mine_output_")


def test_export_raster_writes_geotiff(tmp_path):
    import rasterio

    reference = tmp_path / "ref.tif"
    data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    transform = rasterio.transform.from_origin(10, 20, 1, 1)
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    class RasterTool(DummyExportTool):
        def __init__(self):
            super().__init__()
            self.files_handler = type(
                "Handler",
                (),
                {
                    "band_paths": {"nir": str(reference)},
                    "tif_paths": None,
                },
            )()

    tool = RasterTool()
    tool._output = data * 2

    dest = tool.export_raster(tmp_path, filename="result.tif")

    assert dest.exists()
    with rasterio.open(dest) as src:
        np.testing.assert_array_equal(src.read(1), data * 2)
        assert src.crs.to_epsg() == 4326
        assert src.transform == transform


def test_export_raster_writes_bool_and_multiband(tmp_path):
    import rasterio

    class RasterTool(DummyExportTool):
        def __init__(self):
            super().__init__()
            self.files_handler = type(
                "Handler",
                (),
                {"band_paths": {}, "tif_paths": None},
            )()

    tool = RasterTool()
    tool._output = np.array([[True, False], [False, True]])
    dest = tool.export_raster(tmp_path, filename="mask.tif")
    with rasterio.open(dest) as src:
        assert src.count == 1
        np.testing.assert_array_equal(src.read(1), [[1, 0], [0, 1]])

    tool._output = np.ones((3, 2, 2), dtype=np.float32)
    dest = tool.export_raster(tmp_path, filename="stack.tif")
    with rasterio.open(dest) as src:
        assert src.count == 3
        assert src.read().shape == (3, 2, 2)


def test_export_file_sets_title(tmp_path):
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])
    filename = tool._export_file(tmp_path, title="Demo")
    assert Path(filename).exists()
