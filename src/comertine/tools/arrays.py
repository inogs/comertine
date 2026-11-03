import numpy as np

def enlarge_array(v: np.typing.NDArray, fill_value: float= 0.):
    """
    Define a new array enlarged

    Creates a new array of the same data type and with the same elements as v,
    but each dimension is enlarged with a new element at the beginning and a new
    element at the end. The value of these new entries is `fill_value`.

    Args:
        v: the original array
        fill_value: the value to use to fill the new entries

    Return:
          A new array with 2 more elements for each dimension
    """
    new_shape = tuple(s+ 2 for s in v.shape)
    slices = tuple([slice(1, -1)] * len(v.shape))
    new_v = np.full(new_shape, fill_value, dtype=v.dtype)
    new_v[slices] = v
    return new_v
