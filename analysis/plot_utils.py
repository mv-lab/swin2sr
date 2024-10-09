from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch

from analysis.critic_notebook_utils import get_label_separated_loss, get_mmse_dict
from analysis.lvae_utils import get_img_from_forward_output
from analysis.quantifying_uncertainty import get_regionwise_metric


def clean_ax(ax):
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def add_text(ax, text, img_shape, place='TOP_LEFT'):
    """
    Adding text on image
    """
    assert place in ['TOP_LEFT', 'BOTTOM_RIGHT']
    if place == 'TOP_LEFT':
        ax.text(img_shape[1] * 20 / 500, img_shape[0] * 35 / 500, text, bbox=dict(facecolor='white', alpha=0.9))
    elif place == 'BOTTOM_RIGHT':
        s0 = img_shape[1]
        s1 = img_shape[0]
        ax.text(s0 - s0 * 150 / 500, s1 - s1 * 35 / 500, text, bbox=dict(facecolor='white', alpha=0.9))


def plot_one_batch_twinnoise(imgs, plot_width=20):
    batch_size = len(imgs)
    ncols = batch_size // 2
    img_sz = plot_width // ncols
    _, ax = plt.subplots(figsize=(ncols * img_sz, 2 * img_sz), ncols=ncols, nrows=2)
    for i in range(ncols):
        ax[0, i].imshow(imgs[i, 0])
        ax[1, i].imshow(imgs[i + batch_size // 2, 0])

        ax[1, i].set_title(f'{i + 1 + batch_size // 2}.')
        ax[0, i].set_title(f'{i + 1}.')

        ax[0, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[0, i].axis('off')
        ax[1, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[1, i].axis('off')


def get_k_largest_indices(arr: np.ndarray, K: int):
    """
    Returns the index for K largest elements, in the order small->large.
    """
    ind = np.argpartition(arr, -1 * K)[-1 * K:]
    return ind[np.argsort(arr[ind])]


def add_subplot_axes(ax, rect: List[float], facecolor: str = 'w', min_labelsize: int = 5):
    """
    Add an axes inside an axes. This can be used to create an inset plot.
    Adapted from https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    Args:
        ax: matplotblib.axes
        rect: Array with 4 elements describing where to position the new axes inside the current axes ax.
            eg: [0.1,0.1,0.4,0.2]
        facecolor: what should be the background color of the new axes
        min_labelsize: what should be the minimum labelsize in the new axes
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    # transAxes: co-ordinate system of the axes: 0,0 is bottomleft and 1,1 is top right.
    # With below command, we want to get to a position which would be the position of new plot in the axes coordinate
    # system
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    # with below command, we now have a position of the new plot in the figure coordinate system. we need this because
    # we can create a new axes in the figure coordinate system. so we want to get everything in that system.
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    # subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=max(min_labelsize, x_labelsize))
    subax.yaxis.set_tick_params(labelsize=max(min_labelsize, y_labelsize))
    return subax


def clean_for_xaxis_plot(inset_ax):
    """
    For an xaxis plot, the y axis values don't matter. Neither the axes borders save the bottom one.
    """
    # Removing y-axes ticks and text
    inset_ax.set_yticklabels([])
    inset_ax.tick_params(left=False, right=False)
    inset_ax.set_ylabel('')

    # removing the axes border lines.
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['left'].set_visible(False)


def add_pixel_kde(ax,
                  rect: List[float],
                  data_list: List[np.ndarray],
                  min_labelsize: int,
                  plot_xmax_value: int = None,
                  plot_xmin_value: int = None,
                  plot_kwargs_list=None,
                  color_list=None,
                  label_list=None,
                  color_xtick='white'):
    """
    Adds KDE (density plot) of data1(eg: target) and data2(ex: predicted) image pixel values as an inset
    """
    if plot_kwargs_list is None:
        plot_kwargs_list = [{} for _ in range(len(data_list))]

    inset_ax = add_subplot_axes(ax, rect, facecolor="None", min_labelsize=min_labelsize)

    inset_ax.tick_params(axis='x', colors=color_xtick)
    # xmin, xmax = inset_ax.get_xlim()

    if plot_xmax_value is not None:
        xmax_data = plot_xmax_value
    else:
        xmax_data = [int(datak.max()) for datak in data_list]
        if len(xmax_data) > 1:
            xmax_data = max(*xmax_data) + 1
        else:
            xmax_data = xmax_data[0] + 1

    xmin_data = 0
    if plot_xmin_value is not None:
        xmin_data = plot_xmin_value
    else:
        xmin_data = [int(datak.min()) for datak in data_list]
        if len(xmin_data) > 1:
            xmin_data = min(*xmin_data) - 1
        else:
            xmin_data = xmin_data[0] - 1

    for datak, colork, labelk, plot_kwargsk in zip(data_list, color_list, label_list, plot_kwargs_list):
        sns.kdeplot(data=datak.reshape(-1, ),
                    ax=inset_ax,
                    color=colork,
                    label=labelk,
                    clip=(xmin_data, None),
                    **plot_kwargsk)

    inset_ax.set_aspect('auto')
    inset_ax.set_xlim([xmin_data, xmax_data])  #xmin=0,xmax= xmax_data
    inset_ax.set_xbound(lower=xmin_data, upper=xmax_data)

    xticks = inset_ax.get_xticks()
    inset_ax.set_xticks([xticks[0], xticks[-1]])
    clean_for_xaxis_plot(inset_ax)
    return inset_ax


def plot_imgs_from_idx(idx_list,
                       val_dset,
                       model,
                       model_type,
                       psnr_type='range_invariant',
                       inset_pixel_kde=False,
                       inset_rect=None,
                       inset_min_labelsize=None,
                       color_ch1='red',
                       color_ch2='black',
                       color_generated='pink'):
    """
    Plots  images and their disentangled predictions. Input is a list of idx for which this is done.
    """
    ncols = 5
    nrows = len(idx_list)
    img_sz = 20 / ncols
    _, ax = plt.subplots(figsize=(ncols * img_sz, nrows * img_sz), ncols=ncols, nrows=nrows)

    with torch.no_grad():
        for ax_idx, img_idx in enumerate(idx_list):
            inp, tar = val_dset[img_idx]
            inp = torch.Tensor(inp[None]).cuda()
            tar = torch.Tensor(tar[None]).cuda()

            x_normalized = model.normalize_input(inp)
            target_normalized = model.normalize_target(tar)

            recon_normalized, td_data = model(x_normalized)
            imgs = get_img_from_forward_output(recon_normalized, model)
            loss_dic = get_mmse_dict(model, x_normalized, target_normalized, 1, model_type, psnr_type=psnr_type)
            ll1, ll2 = get_label_separated_loss(loss_dic['mmse_rec_loss'])

            inp = inp.cpu().numpy()
            tar = tar.cpu().numpy()
            imgs = imgs.cpu().numpy()

            psnr1 = loss_dic['psnr_l1'][0]
            psnr2 = loss_dic['psnr_l2'][0]

            ax[ax_idx, 0].imshow(inp[0, 0])
            if inset_pixel_kde:
                # distribution of both labels
                add_pixel_kde(ax[ax_idx, 0],
                              inset_rect,
                              tar[0, 0],
                              tar[0, 1],
                              inset_min_labelsize,
                              label1='Ch1',
                              label2='Ch2',
                              color1=color_ch1,
                              color2=color_ch2)

            # max and min values for label 1
            l1_max = max(tar[0, 0].max(), imgs[0, 0].max())
            l1_min = min(tar[0, 0].min(), imgs[0, 0].min())

            ax[ax_idx, 1].imshow(tar[0, 0], vmin=l1_min, vmax=l1_max)
            ax[ax_idx, 2].imshow(imgs[0, 0], vmin=l1_min, vmax=l1_max)
            add_text(ax[ax_idx, 2], f'PSNR:{psnr1:.1f}', inp.shape[-2:])
            txt = f'{int(l1_min)}-{int(l1_max)}'
            add_text(ax[ax_idx, 2], txt, inp.shape[-2:], place='BOTTOM_RIGHT')
            add_text(ax[ax_idx, 1], txt, inp.shape[-2:], place='BOTTOM_RIGHT')
            if inset_pixel_kde:
                # distribution of label 1 and its prediction
                add_pixel_kde(ax[ax_idx, 2],
                              inset_rect,
                              tar[0, 0],
                              imgs[0, 0],
                              inset_min_labelsize,
                              label1='Ch1',
                              label2='Gen',
                              color1=color_ch1,
                              color2=color_generated)

            # max and min values for label 2
            l2_max = max(tar[0, 1].max(), imgs[0, 1].max())
            l2_min = min(tar[0, 1].min(), imgs[0, 1].min())
            ax[ax_idx, 3].imshow(tar[0, 1], vmin=l2_min, vmax=l2_max)
            ax[ax_idx, 4].imshow(imgs[0, 1], vmin=l2_min, vmax=l2_max)
            txt = f'{int(l2_min)}-{int(l2_max)}'
            add_text(ax[ax_idx, 4], f'PSNR:{psnr2:.1f}', inp.shape[-2:])
            add_text(ax[ax_idx, 4], txt, inp.shape[-2:], place='BOTTOM_RIGHT')
            add_text(ax[ax_idx, 3], txt, inp.shape[-2:], place='BOTTOM_RIGHT')
            if inset_pixel_kde:
                # distribution of label 2 and its prediction
                add_pixel_kde(ax[ax_idx, 4],
                              inset_rect,
                              tar[0, 1],
                              imgs[0, 1],
                              inset_min_labelsize,
                              label1='Ch2',
                              label2='Gen',
                              color1=color_ch2,
                              color2=color_generated)

            ax[ax_idx, 2].set_title(f'Error: {ll1[0]:.3f}')
            ax[ax_idx, 4].set_title(f'Error: {ll2[0]:.3f}')
            ax[ax_idx, 0].set_title(f'Id:{img_idx}')
            ax[ax_idx, 1].set_title('Image 1')
            ax[ax_idx, 3].set_title('Image 2')


def plot_regionwise_metric(model,
                           dset,
                           idx_list: List[int],
                           metric_types: List[str],
                           regionsize: int = 64,
                           sample_count: int = 5,
                           normalize_type=None):
    metric_dict, target = get_regionwise_metric(model,
                                                dset,
                                                idx_list,
                                                metric_types,
                                                regionsize=regionsize,
                                                sample_count=sample_count,
                                                normalize_type=normalize_type)

    img_sz = 3.5
    nrows = len(idx_list)
    sample_count = 20
    inset_rect = [0.1, 0.1, 0.4, 0.2]
    inset_min_labelsize = 8
    _, ax = plt.subplots(figsize=(img_sz * 4, nrows * img_sz), ncols=4, nrows=nrows)
    for i, img_idx in enumerate(idx_list):
        ax[i, 0].imshow(target[img_idx][0])
        ax[i, 2].imshow(target[img_idx][1])

        add_pixel_kde(
            ax[i, 0],
            inset_rect,
            target[img_idx][0],
            target[img_idx][1],
            inset_min_labelsize,
            color1='r',
            color2='black',
        )
        add_pixel_kde(
            ax[i, 2],
            inset_rect,
            target[img_idx][1],
            target[img_idx][0],
            inset_min_labelsize,
            color1='r',
            color2='black',
        )

        max_val = metric_dict[sample_count][img_idx]['RMSE'].max()
        min_val = metric_dict[sample_count][img_idx]['RMSE'].min()
        sns.heatmap(metric_dict[sample_count][img_idx]['RMSE'][0], ax=ax[i, 1], vmax=max_val, vmin=min_val)
        sns.heatmap(metric_dict[sample_count][img_idx]['RMSE'][1], ax=ax[i, 3], vmax=max_val, vmin=min_val)


# Adding arrows.
def add_left_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0] + arrow_length, xy_location[1])
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def add_right_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0] - arrow_length, xy_location[1])
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def add_top_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0], xy_location[1] + arrow_length)
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def add_bottom_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0], xy_location[1] - arrow_length)
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def get_start_vector(xy_start, xy_end, arrow_length):
    """
    Given an arrow_length, return a xy_start such that xy_start => xy_end vector has  this length.
    """
    direction = (xy_end[0] - xy_start[0], xy_end[1] - xy_start[1])
    norm = np.linalg.norm(direction)
    direction = (direction[0] / norm, direction[1] / norm)
    direction = (direction[0] * arrow_length, direction[1] * arrow_length)
    xy_start = (xy_end[0] - direction[0], xy_end[1] - direction[1])
    return xy_start


def add_arrow(ax, xy_start, xy_end, arrow_length=None, color='red', arrowstyle="->"):
    if arrow_length is not None:
        xy_start = get_start_vector(xy_start, xy_end, arrow_length)
    ax.annotate("", xy=xy_end, xytext=xy_start, arrowprops=dict(arrowstyle=arrowstyle, color=color, linewidth=1))
