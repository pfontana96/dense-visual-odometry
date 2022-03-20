# This auxiliary module contains a numpy implementation of some functions for manipulating
# Special Orthogonal and Special Eucledian groups (Lie's algebra)
# It was based on Tim Barfoot's book available on:
# http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf
import abc


_LIE_EPSILON = 1e-6


class BaseSpecialGroup(abc.ABC):
    """
        Interface for Special Groups in Lie's Algebra
    """

    @staticmethod
    @abc.abstractmethod
    def hat(vector):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def exp(vector):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def log(self):
        raise NotImplementedError
