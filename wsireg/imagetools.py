import numpy as np


def scale_by_max(arr, axis=None):
    r"""Divide of every value on the array by max value along an axis.
    .. math::
        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}
    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.
    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios
    Examples
    """
    arr = np.asarray(arr, dtype=float)
    maxval = np.max(arr, axis=axis, keepdims=True)
    return arr / maxval
