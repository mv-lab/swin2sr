import numpy as np


def select_boundary(inp: np.ndarray, width: int):
    """
    Returns the boundary pixels.
    Args:
        inp:numpy.ndarray
        width:

    Returns:

    """
    bnd_pixels = inp.clone()
    bnd_pixels[..., width:-width, width:-width] = np.nan
    filtr = bnd_pixels.isnan()
    bnd_pixels = bnd_pixels[~filtr]

    # checking the sanity. assumption square image.
    pSz = inp.shape[-1]
    pixelcount = 4 * width * pSz - 4 * width * width
    assert pixelcount == np.prod(bnd_pixels.shape)

    return bnd_pixels
