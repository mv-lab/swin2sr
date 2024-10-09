import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import AxesGrid


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    mid_idx = len(reg_index) // 2
    # shifted index to match the data
    shift_index = np.hstack(
        [np.linspace(0.0, midpoint, 128, endpoint=False),
         np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        a = np.abs(ri - reg_index[mid_idx]) / reg_index[mid_idx]
        # print(a)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.register(cmap=newcmap, force=True)

    return newcmap


def get_fractional_change(target, prediction, max_val=None):
    if max_val is None:
        max_val = target.max()
    return (target - prediction) / max_val


def get_zero_centered_midval(error):
    """
    When done this way, the midval ensures that the colorbar is centered at 0. (Don't know how, but it works ;))
    """
    vmax = error.max()
    vmin = error.min()
    midval = 1 - vmax / (vmax + abs(vmin))
    return midval


def plot_error(target, prediction, cmap=matplotlib.cm.coolwarm, ax=None, max_val=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    z2 = get_fractional_change(target, prediction, max_val=max_val)
    midval = get_zero_centered_midval(z2)
    shifted_cmap = shiftedColorMap(cmap, start=0, midpoint=midval, stop=1.0, name='shiftedcmap')
    ax.imshow(prediction, cmap='gray')
    img_err = ax.imshow(z2, cmap=shifted_cmap, alpha=1)
    plt.colorbar(img_err, ax=ax)
