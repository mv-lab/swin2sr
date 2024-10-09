"""
Here, we have functions which can be used to quantify uncertainty in the predictions.
"""
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.lvae_utils import get_img_from_forward_output
from core.psnr import PSNR, RangeInvariantPsnr


def sample_images(model, dset, idx_list, sample_count: int = 5):
    output = {}
    with torch.no_grad():
        for img_idx in idx_list:
            inp, tar = dset[img_idx]
            output[img_idx] = {'rec': [], 'tar': tar}
            inp = torch.Tensor(inp[None]).cuda()
            x_normalized = model.normalize_input(inp)
            for _ in range(sample_count):
                recon_normalized, _ = model(x_normalized)
                imgs = get_img_from_forward_output(recon_normalized, model)
                output[img_idx]['rec'].append(imgs[0].cpu().numpy())

    return output


def compute_regionwise_metric_pairwise_one_pair(data1, data2, metric_types: List[str], regionsize: int):
    # ensure that we are working with a square
    assert data1.shape[-1] == data1.shape[-2]
    assert data1.shape == data2.shape
    Nc = data1.shape[-3]
    Nh = data1.shape[-2] // regionsize
    Nw = data1.shape[-1] // regionsize
    output = {mtype: np.zeros((Nc, Nh, Nw)) for mtype in metric_types}
    for hidx in range(Nh):
        for widx in range(Nw):
            h = hidx * regionsize
            w = widx * regionsize
            d1 = data1[..., h:h + regionsize, w:w + regionsize]
            d2 = data2[..., h:h + regionsize, w:w + regionsize]
            met_dic = _compute_metrics(d1, d2, metric_types)
            for mtype in metric_types:
                output[mtype][..., hidx, widx] = met_dic[mtype]

    return output


def _compute_metrics(data1, data2, metric_types: List[str]):
    data1 = data1.reshape(len(data1), -1)
    data2 = data2.reshape(len(data2), -1)

    output = {}
    #     import pdb;pdb.set_trace()
    for metric_type in metric_types:
        assert metric_type in ['PSNR', 'RangeInvariantPsnr', 'RMSE']

        if metric_type == 'RMSE':
            metric = np.sqrt(np.mean((data1 - data2) ** 2, axis=1))
        elif metric_type == 'PSNR':
            metric = np.array([PSNR(data1[0], data2[0]), PSNR(data1[1], data2[1])])
        elif metric_type == 'RangeInvariantPsnr':
            metric = np.array([RangeInvariantPsnr(data1[0], data2[0]),
                               RangeInvariantPsnr(data1[1], data2[1])])
        output[metric_type] = metric
    return output


def compute_regionwise_metric_pairwise(model, dset, idx_list: List[int], metric_types, regionsize: int = 64,
                                       sample_count: int = 5) -> Dict[int, dict]:
    """
    This will get the prediction multiple times for each of the idx. It would then compute the pairswise metric
    between the predictions, that too on small regions. So, if the model is not sure about a certain region, it would simply
    predict very different things every time and we should get a low PSNR in that region.
    Args:
        model: model
        dset: the dataset
        idx_list: list of idx for which we want to compute this metric
    Returns:
        nested dictionary with following structure img_idx => [pairwise_metric,rec,tar]
                                            pairwise_metric => idx1 => idx2 => metric_type => value
                                            samples => List of sampled reconstructions

    """
    output = {}
    sample_dict = sample_images(model, dset, idx_list, sample_count=sample_count)
    for img_idx in idx_list:
        assert len(sample_dict[img_idx]['rec']) == sample_count
        rec_list = sample_dict[img_idx]['rec']
        output[img_idx] = {'tar': sample_dict[img_idx]['tar'], 'samples': rec_list, 'pairwise_metric': {}}

        for idx1 in range(sample_count):
            output[img_idx]['pairwise_metric'][idx1] = {}
            # NOTE: we need to iterate starting from 0 and not from idx1 + 1 since not every metric is symmetric.
            # PSNR is definitely not.
            for idx2 in range(sample_count):

                if idx1 == idx2:
                    continue
                output[img_idx]['pairwise_metric'][idx1][idx2] = compute_regionwise_metric_pairwise_one_pair(
                    rec_list[idx1],
                    rec_list[idx2],
                    metric_types,
                    regionsize)
    return output


def upscale_regionwise_metric(metric_dict: dict, regionsize: int):
    """
    This expands the regionwise metric to take the same shape as the input image. This ensures that one could simply
    use the heatmap.
    """
    output_dict = {}
    for img_idx in metric_dict.keys():
        output_dict[img_idx] = {}
        for mtype in metric_dict[img_idx].keys():
            metric = metric_dict[img_idx][mtype]
            repeat = np.array([1] * len(metric.shape))
            # The last 2 dimensions are the spatial dimensions. expand it to fit regionsize times the
            # current dimensions.
            repeat[-2:] = regionsize
            metric = np.kron(metric, np.ones(tuple(repeat)))
            output_dict[img_idx][mtype] = metric
    return output_dict


def aggregate_metric(metric_dict):
    """
    Take the average metric over all pairs.
    Args:
        metric_dict: nested dictionary with the following structure.
                    img_idx => pairwise_metric => idx1 => idx2 => metric_type
    Returns:
        aggregated_dict with following structure :img_idx => metric_type
    """
    output_dict = {}
    for img_idx in metric_dict.keys():
        output_dict[img_idx] = {}
        pair_count = 0
        metric_types = []
        for idx1 in metric_dict[img_idx]['pairwise_metric'].keys():
            for idx2 in metric_dict[img_idx]['pairwise_metric'][idx1].keys():
                pair_count += 1
                for metric_type in metric_dict[img_idx]['pairwise_metric'][idx1][idx2]:
                    if metric_type not in output_dict[img_idx]:
                        output_dict[img_idx][metric_type] = 0
                        metric_types.append(metric_type)
                    else:
                        assert metric_type in metric_types

                    output_dict[img_idx][metric_type] += metric_dict[img_idx]['pairwise_metric'][idx1][idx2][
                        metric_type]
        for metric_type in metric_types:
            output_dict[img_idx][metric_type] = output_dict[img_idx][metric_type] / pair_count
    return output_dict


def normalize_metric_single_target(metric_dict: Dict[str, dict], normalize_type: str, target: np.ndarray) -> Dict[
    str, np.ndarray]:
    """
    Args:
        metric_dict: dictionary with the following structure
            metric_type => metric

    """
    assert normalize_type in ['pixelwise_norm']
    normalized_metric = {}
    if normalize_type == 'pixelwise_norm':
        for metric_type in metric_dict:
            metric_mat = metric_dict[metric_type]
            normalized_metric[metric_type] = metric_mat / target
    return normalized_metric


def normalize_metric(metric_dict: Dict[int, dict], normalize_type: str, target_dict: Dict[int, np.ndarray]) -> Dict[
    int, dict]:
    """
    Args:
        metric_dict: nested dictionary with following structure.
                    'img_idx' => 'metric_type' => metric_value
        normalize_type: str
        target_dict: dictionary with following structure.
                'img_idx' => target image.
    """
    normalized_metric_dict = {}
    for img_idx in metric_dict.keys():
        normalized_metric_dict[img_idx] = normalize_metric_single_target(metric_dict[img_idx], normalize_type,
                                                                         target_dict[img_idx])
    return normalized_metric_dict


def get_regionwise_metric(model, dset, idx_list: List[int], metric_types, regionsize: int = 64,
                          sample_count: int = 5, normalize_type='pixelwise_norm'):
    """
    Here, we intend to get regionwise metric. One applies aggregation, upscaling and optionally normalization on top
    of it.
    """
    metric = compute_regionwise_metric_pairwise(model, dset, idx_list, metric_types, regionsize=regionsize,
                                                sample_count=sample_count)
    agg_metric = aggregate_metric(metric)
    target_dict = {img_idx: metric[img_idx]['tar'] for img_idx in metric.keys()}
    upscale_metric = upscale_regionwise_metric(agg_metric, regionsize)

    if normalize_type is not None:
        upscale_metric = normalize_metric(upscale_metric,
                                          normalize_type,
                                          target_dict)

    target = {img_id: metric[img_id]['tar'] for img_id in metric.keys()}
    return upscale_metric, target
