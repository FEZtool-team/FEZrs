from uuid import uuid4
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from rasterio.merge import merge
from rasterio import open as rio_open

from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


class MosaicCalculator(BaseTool):
    def __init__(self, tif_paths: List[BandPathType]):
        super().__init__(tif_paths=tif_paths)
        self.mosaic_rasterio_tifs = self.files_handler.get_rasterio_tifs()

    def _validate(self):
        pass

    def process(self):
        meta = self.mosaic_rasterio_tifs[0].meta.copy()
        mimg, mos_transform = merge(self.mosaic_rasterio_tifs)
        meta.update(
            {
                "driver": "GTiff",
                "height": mimg.shape[1],
                "width": mimg.shape[2],
                "transform": mos_transform,
                "crs": self.mosaic_rasterio_tifs[0].crs,
            }
        )
        self.mosaic_meta = meta
        self.mosaic_mimg = mimg

    def _export_file(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=100,
        bbox_inches="tight",
        grid=False,
    ):
        filename_prefix = self.__class__.__name__.replace("Calculator", "")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        tif_filename = f"{output_path}/{filename_prefix}_{uuid4().hex}.tif"
        with rio_open(tif_filename, "w", **self.mosaic_meta) as dest:
            dest.write(self.mosaic_mimg)
        self._output = tif_filename

        with rio_open(str(self._output)) as src:
            img_data = src.read(1)

        png_filename = f"{output_path}/{filename_prefix}_{uuid4().hex}.png"

        plt.imsave(png_filename, img_data, cmap=colormap, dpi=dpi)

        return

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=100,
        bbox_inches="tight",
        grid=False,
        nrows=None,
        ncols=None,
    ):
        return super().execute(
            output_path,
            title,
            figsize,
            show_axis,
            colormap,
            show_colorbar,
            filename_prefix,
            dpi,
            bbox_inches,
            grid,
            nrows,
            ncols,
        )
