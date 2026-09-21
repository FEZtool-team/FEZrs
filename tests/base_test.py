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


def test_process_runs_validate_without_execute():
    """
    Issue #66: every calculator overrides process() and almost none called
    _validate(), so process() then to_raster() skipped the guards execute()
    runs. BaseTool wraps subclass process() so validation is not path-dependent.
    """
    tool = DummyTool()

    tool.process()

    assert tool._validate_called is True
    assert tool._process_called is True


def test_process_wrapper_is_on_every_public_calculator():
    import inspect

    import fezrs

    missing = []
    for name in fezrs.__all__:
        cls = getattr(fezrs, name)
        if not inspect.isclass(cls) or not issubclass(cls, BaseTool):
            continue
        if cls is BaseTool:
            continue
        if "process" in cls.__dict__ and not getattr(
            cls.process, "_fezrs_runs_validate", False
        ):
            missing.append(name)

    assert missing == []


def test_execute_returns_self():
    tool = DummyTool()

    result = tool.execute(".")

    assert result is tool


# --- Export arguments that were accepted and then ignored (issue #42) ---------


def test_export_file_honours_filename_prefix(tmp_path):
    """
    filename_prefix was accepted by all 37 calculators, forwarded correctly, and
    then overwritten on the first line of _export_file() before it was used.
    """
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])

    filename = tool._export_file(tmp_path, filename_prefix="my_run")

    assert Path(filename).name.startswith("my_run_output_")


def test_export_file_defaults_to_the_tool_name(tmp_path):
    """Omitting the prefix must keep the previous default filenames."""
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])

    filename = tool._export_file(tmp_path)

    assert Path(filename).name.startswith("Dummy_output_")


def test_execute_forwards_nrows_and_ncols():
    """
    nrows/ncols were declared and documented on execute() but never reached
    _export_file(), so a multi-panel layout was silently ignored.
    """
    captured = {}

    class RecordingTool(BaseTool):
        def __init__(self):
            self._output = np.array([[1]])

        def _validate(self):
            pass

        def process(self):
            pass

        def _export_file(self, *args, **kwargs):
            captured["args"] = args
            return "out.png"

    RecordingTool().execute(".", nrows=2, ncols=3)

    # positional tail of _export_file(...) is (..., grid, nrows, ncols)
    assert captured["args"][-2:] == (2, 3)


def test_execute_defaults_nrows_and_ncols_to_single_panel():
    captured = {}

    class RecordingTool(BaseTool):
        def __init__(self):
            self._output = np.array([[1]])

        def _validate(self):
            pass

        def process(self):
            pass

        def _export_file(self, *args, **kwargs):
            captured["args"] = args
            return "out.png"

    RecordingTool().execute(".")

    assert captured["args"][-2:] == (1, 1)


def test_export_file_produces_the_requested_grid(tmp_path):
    tool = DummyExportTool()
    tool._output = np.array([[1, 2], [3, 4]])

    filename = tool._export_file(tmp_path, nrows=1, ncols=1)

    assert Path(filename).is_file()
