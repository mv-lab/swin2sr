import numpy as np
from skimage.io import imread, imsave


def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin='tifffile')
    return data


def save_tiff(path, data):
    imsave(path, data, plugin='tifffile')


def load_tiffs(paths):
    data = [load_tiff(path) for path in paths]
    return np.concatenate(data, axis=0)
