import ctypes as C

from numba import cuda
import numpy as np
import numpy.typing as npt


class UnifiedMemoryArray:
    """Class for allocating arrays in UVM memory.
    """

    _NUMPY_TO_C_DTYPES = {
        np.uint8: C.c_uint8,
        np.uint16: C.c_uint16,
        np.float32: C.c_float,
        bool: C.c_bool
    }

    def __init__(self, shape: tuple, dtype: npt.DTypeLike, data: npt.ArrayLike = None):
        """Creates an array accesible from both GPU and CPU to avoid data copying.

        Parameters
        ----------
        shape : tuple
            Shape of the array to allocate.
        data_dtype : npt.DTypeLike
            Numpy's data type of the array to allocate.
        data : npt.ArrayLike, optional
            Numpy array containing data (on the host) to initialize, by default None.
        """
        self._shape = shape
        self._np_dtype = dtype

        self._gpu = cuda.managed_array(self._shape, dtype=self._np_dtype, strides=None, order='C', stream=0)
        self._cpu = np.ctypeslib.as_array(
            C.cast(self._gpu.ctypes.data, C.POINTER(self._get_c_type(self._np_dtype))), shape=self._shape
        )

        if data is not None:

            if data.shape != self._shape:
                raise ValueError("Expected 'data' shape to match defined one '{}', got '{}' instead".format(
                    self._shape, data.shape
                ))

            if data.dtype != self._np_dtype:
                raise ValueError("Expected 'data' dtype to match defined one '{}', got '{}' instead".format(
                    self._np_dtype, data.dtype
                ))

            self._cpu[...] = data

    @staticmethod
    def _get_c_type(np_dtype: npt.DTypeLike):
        try:
            result = UnifiedMemoryArray._NUMPY_TO_C_DTYPES[np_dtype]

        except KeyError:
            raise ValueError("No conversion from numpy data type '{}' to a C one found. Valid conversions '{}'".format(
                np_dtype, list(UnifiedMemoryArray._NUMPY_TO_C_DTYPES.keys())
            ))

        return result

    def get(self, device: str) -> npt.ArrayLike:
        """Returns the array

        Parameters
        ----------
        device : str
            Device to get the array ('gpu' or 'cpu')

        Returns
        -------
        Union[NDManagedArray, NDArray]
            Array on the host or GPU (same physical memory address)
        """
        device = device.lower()
        assert device in ["cpu", "gpu"], "Expected 'device' to be either 'gpu' or 'cpu', got '{}'".format(
            device
        )

        if device == "cpu":
            return self._cpu

        if device == "gpu":
            return self._gpu
