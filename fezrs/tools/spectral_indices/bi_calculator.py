import warnings

# Import module and files
from fezrs.base import BaseTool
from fezrs.tools.spectral_indices._division import divide_with_nan
from fezrs.utils.radiometry_handler import apply_scaling
from fezrs.utils.type_handler import BandPathType, BIFormulationType


# Calculator class
class BICalculator(BaseTool):
    """
    Bare soil / exposed rock index.

    Two formulations are available.

    ``"bsi"`` (default when SWIR1 and Blue are supplied) is the Bare Soil Index:

        BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))

    It works because bare soil and exposed rock are bright in SWIR1 and Red and
    dark in NIR and Blue, which is the inverse of the vegetation response. That
    four-band contrast is what separates lithological exposure from canopy, and
    it is the standard instrument for the target this tool documents.

    ``"legacy"`` is the expression previous releases computed:

        BI_legacy = (NIR - Green - Red) / (NIR + Green + Red)

    It is retained so existing workflows keep running and remain reproducible,
    but it is not a published bare-soil index: it subtracts two visible bands
    from NIR, so it responds primarily to vegetation brightness rather than to
    soil or rock exposure. Prefer ``"bsi"`` for new work.
    """

    def __init__(
        self,
        nir_path: BandPathType,
        red_path: BandPathType,
        green_path: BandPathType | None = None,
        swir1_path: BandPathType | None = None,
        blue_path: BandPathType | None = None,
        formulation: BIFormulationType | None = None,
        scale_factor: float = 1.0,
        offset: float = 0.0,
    ):
        """
        Args:
            nir_path: Near-infrared band.
            red_path: Red band.
            green_path: Green band, required by the ``"legacy"`` formulation.
            swir1_path: SWIR ~1.6 um band, required by ``"bsi"``.
            blue_path: Blue band, required by ``"bsi"``.
            formulation: ``"bsi"`` or ``"legacy"``. Inferred from the supplied
                bands when omitted.
            scale_factor: Multiplicative radiometric scale, see RADIOMETRIC_PRESETS.
            offset: Additive radiometric offset.
        """
        if formulation is None:
            formulation = (
                "bsi" if swir1_path is not None and blue_path is not None else "legacy"
            )

        if formulation not in ("bsi", "legacy"):
            raise ValueError(
                f"Invalid BI formulation: {formulation!r}. "
                "Must be 'bsi' or 'legacy'."
            )

        self.formulation: BIFormulationType = formulation

        if formulation == "bsi":
            if swir1_path is None or blue_path is None:
                raise ValueError(
                    "The 'bsi' formulation requires both swir1_path and blue_path."
                )
            self._required_bands = ("swir1", "red", "nir", "blue")
        else:
            if green_path is None:
                raise ValueError(
                    "The 'legacy' formulation requires green_path."
                )
            warnings.warn(
                "BICalculator's 'legacy' formulation "
                "(NIR - Green - Red) / (NIR + Green + Red) is not a published "
                "bare-soil index and responds mainly to vegetation brightness. "
                "Pass swir1_path and blue_path to compute BSI instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._required_bands = ("nir", "red", "green")

        band_paths = {"nir_path": nir_path, "red_path": red_path}
        if green_path is not None:
            band_paths["green_path"] = green_path
        if swir1_path is not None:
            band_paths["swir1_path"] = swir1_path
        if blue_path is not None:
            band_paths["blue_path"] = blue_path

        super().__init__(**band_paths)

        self.source_bands = apply_scaling(
            self.files_handler.get_bands(requested_bands=list(self._required_bands)),
            scale_factor=scale_factor,
            offset=offset,
        )

    def _validate(self):
        pass

    def process(self):
        if self.formulation == "bsi":
            swir1, red, nir, blue = (
                self.source_bands[band]
                for band in ("swir1", "red", "nir", "blue")
            )
            self._output = divide_with_nan(
                (swir1 + red) - (nir + blue),
                (swir1 + red) + (nir + blue),
            )
        else:
            nir, red, green = (
                self.source_bands[band] for band in ("nir", "red", "green")
            )
            self._output = divide_with_nan(
                (nir - green) - red,
                (nir + green) + red,
            )

        return self._output

    def execute(
        self,
        output_path,
        title=None,
        figsize=(15, 10),
        show_axis=False,
        colormap="gray",
        show_colorbar=True,
        filename_prefix=None,
        dpi=1000,
        bbox_inches="tight",
        grid=True,
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
