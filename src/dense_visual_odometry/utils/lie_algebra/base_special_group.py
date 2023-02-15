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

    @abc.abstractmethod
    def hat(vector):
        pass

    @abc.abstractmethod
    def exp(vector):
        pass

    @abc.abstractmethod
    def log(self):
        pass

    @abc.abstractmethod
    def inverse(self):
        pass

    @abc.abstractmethod
    def __mul__(self, right):
        pass

    @abc.abstractmethod
    def __eq__(self, right):
        pass
