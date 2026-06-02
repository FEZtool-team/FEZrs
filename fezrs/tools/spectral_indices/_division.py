import numpy as np


def divide_with_nan(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.result_type(numerator, float))
    return np.divide(numerator, denominator, out=result, where=denominator != 0)
