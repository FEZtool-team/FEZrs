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
