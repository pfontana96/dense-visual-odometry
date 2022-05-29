from functools import lru_cache, wraps
import numpy as np
from typing import Callable


def np_cache(function: Callable):
    """
        Decorator to make numpy function cacheable (numpy arrays are not hashable).
        See: https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays
    """
    @lru_cache(maxsize=None)
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # Copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
