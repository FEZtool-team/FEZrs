"""
Radiometric scaling for spectral indices.

Satellite products are distributed as scaled integers, not as reflectance. A
spectral index built from ratios of *the same* linear scale is insensitive to a
common gain, but two classes of index are not:

* indices carrying an **additive constant** in reflectance units, such as SAVI's
  soil adjustment ``L = 0.5``;
* indices carrying a **band-specific coefficient**, such as AFRI's 0.66 and 0.50.

For those the input must actually be reflectance, otherwise the constant is
being added to a digital number in the thousands and contributes nothing. This
module converts a product's stored integers to reflectance:

    rho = DN * scale_factor + offset
"""

import warnings

import numpy as np


# Published scaling for the common analysis-ready products. Splat one into a
# calculator: NDVICalculator(..., **RADIOMETRIC_PRESETS["landsat-c2-l2"]).
RADIOMETRIC_PRESETS = {
    # Landsat Collection 2 Level-2 surface reflectance.
    "landsat-c2-l2": {"scale_factor": 2.75e-5, "offset": -0.2},
    # Sentinel-2 L2A, processing baseline < 04.00.
    "sentinel2-l2a": {"scale_factor": 1e-4, "offset": 0.0},
    # Sentinel-2 L2A, processing baseline >= 04.00 carries BOA_ADD_OFFSET.
    "sentinel2-l2a-baseline4": {"scale_factor": 1e-4, "offset": -0.1},
    # Input already expressed as reflectance.
    "reflectance": {"scale_factor": 1.0, "offset": 0.0},
}

# Physically plausible reflectance bounds, with headroom for specular returns
# and for the slight out-of-range values common in atmospherically corrected
# products.
_REFLECTANCE_CEILING = 1.5

# Indices whose constants are defined in reflectance units, so feeding them
# unscaled digital numbers silently produces a meaningless result.
_REFLECTANCE_DEPENDENT = ("SAVI", "AFRI")


def apply_scaling(bands, scale_factor=1.0, offset=0.0):
    """
    Convert stored band values to reflectance.

    Args:
        bands: Mapping of band name to array.
        scale_factor: Multiplicative scale from the product specification.
        offset: Additive offset from the product specification.

    Returns:
        dict: Mapping of band name to scaled array. Returned unchanged when the
            scaling is the identity, so the common case allocates nothing.
    """
    if scale_factor == 1.0 and offset == 0.0:
        return dict(bands)

    return {
        name: None if array is None else np.asarray(array, dtype=float) * scale_factor
        + offset
        for name, array in bands.items()
    }


def warn_if_not_reflectance(bands, index_name):
    """
    Warn when a reflectance-dependent index is handed unscaled data.

    Args:
        bands: Mapping of band name to array.
        index_name: Index label, used to decide whether the check applies and to
            build the message.
    """
    if index_name not in _REFLECTANCE_DEPENDENT:
        return

    for name, array in bands.items():
        if array is None:
            continue

        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size == 0:
            continue

        peak = float(np.max(np.abs(finite)))
        if peak > _REFLECTANCE_CEILING:
            warnings.warn(
                f"{index_name} band {name!r} reaches {peak:.1f}, which is far "
                "outside the reflectance range [0, 1]. This index carries "
                "constants defined in reflectance units, so unscaled digital "
                "numbers produce a value with no physical meaning. Pass "
                "scale_factor/offset for your product, e.g. "
                "**RADIOMETRIC_PRESETS['landsat-c2-l2'].",
                UserWarning,
                stacklevel=3,
            )
            return
