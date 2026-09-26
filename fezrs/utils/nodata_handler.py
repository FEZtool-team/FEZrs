"""
Nodata handling shared by the calculators.

A declared nodata fill is a mask, not a value. ``FileHandler`` already turns the
fill into NaN when it reads a band that declares one, but that alone is not
enough: many of the algorithms the calculators wrap cannot take NaN. OpenCV's
median filter rejects float64 outright, scikit-image's histogram equalisation
cannot build a histogram of NaN, and scikit-learn's KMeans and SVC refuse
missing values. Left there, masking the fill trades a silently wrong answer for
a crash on the most ordinary input there is: a scene with a fill margin.

The rule every calculator follows:

* **Inputs without declared nodata are untouched.** No mask is built and the
  output is exactly what it was before.
* **When inputs declare nodata, the output is float with NaN over it.** The
  computation runs on valid pixels only, and the fill comes back as NaN rather
  than as a number that reads like a measurement (an NDVI of 0, a burn flag of
  False, a cluster label).
* **Windowed operations also report NaN wherever the window touches nodata.**
  A filtered, gradient or texture value computed partly from fill is not a
  measurement of the surface, and at a scene edge it manufactures a false
  boundary -- a texture contrast spike along the fill line looks exactly like a
  lithological contact.
"""

from typing import Optional

import cv2
import numpy as np


def invalid_mask(*arrays) -> Optional[np.ndarray]:
    """
    Pixels that are nodata in any of the given arrays.

    Every array must share the same leading ``(height, width)``; a trailing
    channel axis, as on an RGB composite, counts as invalid if any channel is.

    Returns:
        A boolean ``(height, width)`` mask, or ``None`` when every pixel of every
        array is valid. ``None`` is the signal to leave the computation and its
        output completely unchanged.
    """
    mask = None
    for array in arrays:
        if array is None:
            continue
        values = np.asarray(array)
        if not np.issubdtype(values.dtype, np.floating):
            # Integer or boolean data cannot hold NaN, so it has no nodata.
            continue
        bad = ~np.isfinite(values)
        if bad.ndim == 3:
            bad = bad.any(axis=2)
        mask = bad if mask is None else (mask | bad)

    if mask is None or not mask.any():
        return None
    return mask


def dilate(mask: np.ndarray, window: int) -> np.ndarray:
    """
    Grow a mask by the footprint of a ``window`` x ``window`` operation.

    A pixel whose window reaches any masked pixel is itself masked.
    """
    if window <= 1:
        return mask
    kernel = np.ones((window, window), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def fill_invalid(array: np.ndarray, mask: Optional[np.ndarray], value=None) -> np.ndarray:
    """
    A copy of ``array`` with masked pixels replaced by a finite value.

    For algorithms that cannot take NaN at all. The filled pixels are masked out
    of the result again afterwards, so the value only has to keep the algorithm
    well behaved. It defaults to the median of the valid pixels: unlike 0 or the
    minimum, the median does not drag a histogram or a local statistic towards
    an extreme.
    """
    values = np.asarray(array, dtype=np.float64)
    if mask is None:
        return values

    valid = values[~mask] if values.ndim == 2 else values[~mask].reshape(-1)
    if value is None:
        finite = valid[np.isfinite(valid)]
        value = float(np.median(finite)) if finite.size else 0.0

    filled = values.copy()
    filled[mask] = value
    return filled


def apply_nodata(output, mask: Optional[np.ndarray], window: int = 1):
    """
    Write NaN over the masked pixels of ``output``.

    Args:
        output: The computed result, ``(height, width)`` or
            ``(height, width, channels)``.
        mask: From :func:`invalid_mask`. ``None`` returns ``output`` unchanged.
        window: For windowed operations, the window edge length; the mask is
            dilated by it first, so any pixel whose window touched nodata is
            masked too.

    Returns:
        ``output`` promoted to float with NaN over nodata, or ``output`` itself
        when there is no mask.
    """
    if mask is None:
        return output

    result = np.array(output, dtype=np.float64, copy=True)
    grown = dilate(mask, window)

    if result.ndim == 3:
        result[grown, :] = np.nan
    else:
        result[grown] = np.nan
    return result
