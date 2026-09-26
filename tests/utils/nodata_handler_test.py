import numpy as np

from fezrs.utils.nodata_handler import (
    apply_nodata,
    dilate,
    fill_invalid,
    invalid_mask,
)


def test_no_mask_when_every_pixel_is_valid():
    """None is the signal to leave a computation completely untouched."""
    assert invalid_mask(np.ones((3, 3))) is None


def test_integer_and_boolean_data_have_no_nodata():
    assert invalid_mask(np.arange(9).reshape(3, 3)) is None
    assert invalid_mask(np.zeros((3, 3), dtype=bool)) is None


def test_mask_is_the_union_across_arrays_and_channels():
    a = np.ones((3, 3))
    b = np.ones((3, 3, 2))
    a[0, 0] = np.nan
    b[2, 2, 1] = np.nan

    mask = invalid_mask(a, b, None)

    assert mask.shape == (3, 3)
    assert mask[0, 0] and mask[2, 2]
    assert mask.sum() == 2


def test_dilation_covers_the_window_footprint():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    assert dilate(mask, 1).sum() == 1
    assert dilate(mask, 3).sum() == 9


def test_fill_defaults_to_the_valid_median():
    """The median, unlike 0 or the minimum, does not drag a statistic to an extreme."""
    array = np.array([[1.0, 2.0], [9.0, np.nan]])

    filled = fill_invalid(array, invalid_mask(array))

    assert filled[1, 1] == 2.0
    assert np.isfinite(filled).all()


def test_apply_nodata_is_a_no_op_without_a_mask():
    output = np.array([[True, False]])

    assert apply_nodata(output, None) is output


def test_apply_nodata_promotes_and_masks():
    mask = np.array([[True, False], [False, False]])

    result = apply_nodata(np.array([[True, False], [True, True]]), mask)

    assert result.dtype == np.float64
    assert np.isnan(result[0, 0])
    assert result[1, 0] == 1.0


def test_apply_nodata_masks_every_channel():
    mask = np.zeros((2, 2), dtype=bool)
    mask[1, 1] = True

    result = apply_nodata(np.ones((2, 2, 3)), mask)

    assert np.isnan(result[1, 1, :]).all()
    assert np.isfinite(result[0, 0, :]).all()
