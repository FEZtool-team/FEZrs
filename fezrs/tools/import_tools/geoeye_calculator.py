from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


class Geoeye_Calculator(BaseTool):
    def __init__(self, tif_path: BandPathType, level: int = 0):
        super().__init__(
            tif_path=tif_path,
        )

        self.tif_normalized = self.files_handler.get_normalized_bands(
            requested_bands=["tif"]
        )

        self.level = level

    def _validate(self):
        number_of_bands = self.tif_normalized["tif"].shape[2]

        if not (0 <= self.level < number_of_bands):
            raise ValueError(
                f"Invalid level {self.level}. It must be between 0 and {number_of_bands - 1}."
            )

    def process(self):
        self.tif_normalize_level = self.tif_normalized["tif"][:, :, self.level]
        self._output = self.tif_normalize_level

    def execute(
        self,
        output_path,
        title=None,
        figsize=(10, 10),
        show_axis=True,
        colormap="gray",
        show_colorbar=False,
        filename_prefix="Tool_output",
        dpi=500,
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
