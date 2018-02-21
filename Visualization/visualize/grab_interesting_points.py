import numpy as np
from visualize.basic_visualizer import GrabInterestPointsInterface
from interface import implements


class GrabMaximumPoints(implements(GrabInterestPointsInterface)):
    def __init__(self, maximum_percentage=10):
        self.maximum_percentage=maximum_percentage

    """
    Grab maximum XX% values
    """
    def find_indices(self, D, intensity):
        shape = np.shape(D)
        if len(shape) != 3:
            raise ValueError("Invalid shape of input array. Should be 4-D tensor with last dimension as channels.")

        D_flatten = D.flatten()
        sorted_D_flatten = np.sort(D_flatten)
        len_D = len(D_flatten)

        # (n-1) would be the maximum value
        # We need len_D * self.maximum_percentage / 100 values
        percentage = intensity * self.maximum_percentage / 100.0
        print("Percentage", percentage, len_D * percentage / 100)
        required_len = int(len_D * percentage / 100)
        if required_len == 0:
            return None
        threshold = sorted_D_flatten[len_D - required_len]

        return np.argwhere(D >= threshold)


