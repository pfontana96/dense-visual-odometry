from functools import lru_cache, wraps
import numpy as np
from typing import Callable


def np_cache(function: Callable):
    """
        Decorator to make numpy function cacheable (numpy arrays are not hashable). It only works for functions
        containing either `*args` or `**kwargs` being 2D numpy arrays and won't work for arrays with different
        dimensions. See: https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays
    """
    @lru_cache(maxsize=10, typed=False)
    def cached_wrapper(*args, **kwargs):
        np_args_index = kwargs.pop("numpy_args_index")
        np_kwargs_keys = kwargs.pop("numpy_args_keys")

        args = recover_numpy_args(np_args_index, *args)
        kwargs = recover_numpy_kwargs(np_kwargs_keys, **kwargs)

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args, np_args_index = hash_numpy_args(*args)
        kwargs, np_args_keys = hash_numpy_kwargs(**kwargs)
        kwargs.update({
            "numpy_args_index": np_args_index,
            "numpy_args_keys": np_args_keys
        })

        return cached_wrapper(*args, **kwargs)

    def hash_numpy_args(*args):
        result = tuple()
        np_args_index = tuple()
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                arg = tuple(map(tuple, arg))
                np_args_index += (i,)

            result += (arg,)

        return result, np_args_index

    def recover_numpy_args(np_args_index, *args):
        args = list(args)
        for index in np_args_index:
            args[index] = np.array(args[index])

        return args

    def hash_numpy_kwargs(**kwargs):
        result = {}
        np_kwargs_keys = tuple()
        for key, arg in kwargs.items():
            if isinstance(arg, np.ndarray):
                arg = tuple(map(tuple, arg))
                np_kwargs_keys += (key,)
            result.update({key: arg})

        return result, np_kwargs_keys

    def recover_numpy_kwargs(np_kwargs_keys, **kwargs):
        for key in np_kwargs_keys:
            kwargs[key] = np.array(kwargs[key])

        return kwargs

    # Copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
