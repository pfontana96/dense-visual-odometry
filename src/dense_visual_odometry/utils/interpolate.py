import numpy as np


class Interp2D:
    """
        Class with different 2D interpolation methods
    """

    @staticmethod
    def bilinear(x: np.ndarray, y: np.ndarray, image: np.ndarray):
        """
            Bilinear Interpolation

        Parameters
        ----------
        x : np.ndarray
            Array of the x pixel coordinates values to interpolate from `ìmage`
        y : np.ndarray
            Array of the y pixel coordinates values to interpolate from `ìmage`
        image : np.ndarrray
            Image to interpolate values from

        Returns
        -------
        values : np.ndarray
            Interpolated values for the coordinates specified by `x` and `y` from `image`
        """
        assert len(x) == len(y), "'x' and 'y' should have the same length"

        height, width = image.shape

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)

        x1 = x0 + 1
        y1 = y0 + 1

        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)

        Ia = image[y0, x0].astype(np.float32)
        Ib = image[y1, x0].astype(np.float32)
        Ic = image[y0, x1].astype(np.float32)
        Id = image[y1, x1].astype(np.float32)

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return (wa * Ia + wb * Ib + wc * Ic + wd * Id)
