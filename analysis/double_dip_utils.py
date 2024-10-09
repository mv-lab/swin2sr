import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_utils import clean_ax
from core.psnr import RangeInvariantPsnr


def get_psnr(gt, pred):
    """
    Order in the prediction is not fixed. So, we  compute the psnr of each ground truth with both predictions
    and then pick the correct ordering based on the psnr value.
    """
    psnr0_0 = RangeInvariantPsnr(gt[0], pred[0])
    psnr0_1 = RangeInvariantPsnr(gt[0], pred[1])

    psnr1_0 = RangeInvariantPsnr(gt[1], pred[0])
    psnr1_1 = RangeInvariantPsnr(gt[1], pred[1])
    if psnr0_0 + psnr1_1 > psnr0_1 + psnr1_0:
        return psnr0_0, psnr1_1
    else:
        return psnr0_1, psnr1_0


def step_num(fname: str) -> int:
    """
    sum1_499.jpg => 499
    """
    return int(fname.split('.')[0].split('_')[-1])


def get_fpath_sequence(prefix, rootdir, extension=None):
    """
    Args:
        prefix: file name should start with prefix
        rootdir:
        extension:str
    """
    output = []
    for fname in os.listdir(rootdir):
        if prefix == fname[:len(prefix)]:
            if extension is not None:
                if fname[-1 * len(extension):] != extension:
                    continue

            output.append(os.path.join(rootdir, fname))

    return sorted(output, key=lambda x: step_num(os.path.basename(x)))


def show_imgs_from_np_fpaths(fpath_list, ncols=4, img_sz=5, title_list=None, preprocessing_fn=None):
    nrows = int(np.ceil(len(fpath_list) / ncols))
    _, ax = plt.subplots(figsize=(img_sz * ncols, nrows * img_sz), ncols=ncols, nrows=nrows)
    clean_ax(ax)
    if len(ax.shape) == 1:
        ax = ax.reshape(1, -1)
    for ridx in range(nrows):
        for cidx in range(ncols):
            fpath_idx = ridx * nrows + cidx
            fpath = fpath_list[fpath_idx]
            img = np.load(fpath)
            if preprocessing_fn is not None:
                img = preprocessing_fn(img)

            ax[ridx, cidx].imshow(img[0])
            if isinstance(title_list, list):
                title = title_list[fpath_idx]
            ax[ridx, cidx].set_title(title)
