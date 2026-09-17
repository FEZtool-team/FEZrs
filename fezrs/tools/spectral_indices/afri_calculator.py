# Import module and files
from fezrs.base import BaseTool
from fezrs.tools.spectral_indices._division import divide_with_nan
from fezrs.utils.radiometry_handler import apply_scaling, warn_if_not_reflectance
from fezrs.utils.type_handler import AFRIVariantType, BandPathType


# Coefficients from Karnieli et al. (2001), doi:10.1016/S0034-4257(01)00190-0.
# Each derives from an empirical relationship between visible and SWIR
# reflectance over vegetated surfaces: rho_0.645 ~= 0.66 * rho_1.6 and
# rho_0.469 ~= 0.50 * rho_2.1. The coefficient only has meaning as a scaling
# factor applied to the SWIR band.
AFRI_COEFFICIENTS = {"1.6": 0.66, "2.1": 0.50}

AFRI_BANDS = {"1.6": "swir1", "2.1": "swir2"}


# Calculator class
class AFRICalculator(BaseTool):
    """
    Aerosol Free Vegetation Index (Karnieli et al., 2001).

    AFRI substitutes a SWIR band for the visible red used by NDVI. Aerosol
    scattering is strongly wavelength dependent and falls off toward longer
    wavelengths, so a SWIR-based index penetrates smoke, haze and dust that
    would otherwise depress a red-based index. Under clear-sky conditions
    Karnieli et al. report that AFRI and NDVI are almost identical, which is a
    useful check on any implementation.

    Two formulations are defined:

        AFRI_1.6 = (NIR - 0.66 * SWIR1.6) / (NIR + 0.66 * SWIR1.6)
        AFRI_2.1 = (NIR - 0.50 * SWIR2.1) / (NIR + 0.50 * SWIR2.1)
    """

    def __init__(
        self,
        nir_path: BandPathType,
        swir1_path: BandPathType | None = None,
        swir2_path: BandPathType | None = None,
        variant: AFRIVariantType = "1.6",
        scale_factor: float = 1.0,
        offset: float = 0.0,
    ):
        """
        Args:
            nir_path: Near-infrared band.
            swir1_path: SWIR ~1.6 um band, required for the ``"1.6"`` variant.
            swir2_path: SWIR ~2.1 um band, required for the ``"2.1"`` variant.
            variant: Which AFRI formulation to compute.
            scale_factor: Multiplicative radiometric scale, see RADIOMETRIC_PRESETS.
            offset: Additive radiometric offset.
        """
        if variant not in AFRI_COEFFICIENTS:
            raise ValueError(
                f"Invalid AFRI variant: {variant!r}. "
                f"Must be one of {sorted(AFRI_COEFFICIENTS)}."
            )

        self.variant: AFRIVariantType = variant
        self.coefficient = AFRI_COEFFICIENTS[variant]
        self.swir_band = AFRI_BANDS[variant]

        required_path = swir1_path if variant == "1.6" else swir2_path
        if required_path is None:
            raise ValueError(
                f"AFRI variant {variant!r} requires "
                f"{'swir1_path' if variant == '1.6' else 'swir2_path'}."
            )

        band_paths = {"nir_path": nir_path}
        if swir1_path is not None:
            band_paths["swir1_path"] = swir1_path
        if swir2_path is not None:
            band_paths["swir2_path"] = swir2_path

        super().__init__(**band_paths)

        self.source_bands = apply_scaling(
            self.files_handler.get_bands(requested_bands=["nir", self.swir_band]),
            scale_factor=scale_factor,
            offset=offset,
        )
        warn_if_not_reflectance(self.source_bands, "AFRI")

    def _validate(self):
        pass

    def process(self):
        nir, swir = (
            self.source_bands[band] for band in ("nir", self.swir_band)
        )

        self._output = divide_with_nan(
            nir - self.coefficient * swir,
            nir + self.coefficient * swir,
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
