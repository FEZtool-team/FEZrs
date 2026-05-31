import numpy as np
import matplotlib.pyplot as plt

from fezrs.utils.histogram_handler import HistogramExportMixin


class DummyHistogramExport(HistogramExportMixin):
    def __init__(self):
        self._logo_watermark = np.zeros((10, 10, 3))


def test_add_watermark_adds_artist():
    obj = DummyHistogramExport()
    fig, ax = plt.subplots()

    initial_count = len(ax.artists)

    obj._add_watermark(ax)

    assert len(ax.artists) == initial_count + 1

    plt.close(fig)


def test_save_histogram_figure_saves_file_and_returns_path(tmp_path):
    obj = DummyHistogramExport()
    fig, ax = plt.subplots()

    output_file = obj._save_histogram_figure(
        ax=ax,
        output_path=tmp_path,
        filename_prefix="histogram",
        dpi=100,
        bbox_inches="tight",
    )

    assert output_file.endswith(".png")
    assert "histogram_" in output_file
    assert tmp_path.joinpath(output_file.split("/")[-1]).exists()


def test_save_histogram_figure_closes_figure(tmp_path):
    obj = DummyHistogramExport()
    fig, ax = plt.subplots()

    fig_number = fig.number

    obj._save_histogram_figure(
        ax=ax,
        output_path=tmp_path,
        filename_prefix="histogram",
        dpi=100,
        bbox_inches="tight",
    )

    assert fig_number not in plt.get_fignums()
