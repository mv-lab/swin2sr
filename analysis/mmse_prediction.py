from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.loss_type import LossType
from core.model_type import ModelType
from metrics.running_psnr import RunningPSNR


def get_mmse_prediction(model, dset, inp_idx, mmse_count, padded_size: int, prediction_size: int, batch_size=16,
                        track_progress: bool = True) -> \
        Tuple[
            torch.Tensor, torch.Tensor]:
    """
    The work here is to simply get the MMSE prediction for a specific input.
    Args:
        model:
        dset: the dataset.
        inp_idx: Input index of the dataset for which MMSE prediction needs to be computed.
        mmse_count: Averaging over how many times?
        padded_size: After padding what should be size of the input to the model.
        prediction_size: How much should be kept for prediction. Ex: padded_size=96 and prediction_size=64. 16 pixesls
                        are padded on all sides in this case.
        batch_size: Used for speeding up the computation.

    Returns:
        MMSE prediction and the target. Both are in normalized state.

    """
    assert padded_size >= prediction_size
    old_img_sz = dset.get_img_sz()
    dset.set_img_sz(padded_size)

    padN = (padded_size - prediction_size) // 2

    with torch.no_grad():
        inp, tar = dset[inp_idx]
        inp = torch.Tensor(inp[None])
        tar = torch.Tensor(tar[None])
        inp = inp.repeat(batch_size, 1, 1, 1)
        tar = tar.repeat(batch_size, 1, 1, 1)
        inp = inp.cuda()
        tar = tar.cuda()
        recon_img_list = []
        range_mmse = range(0, mmse_count, batch_size)
        if track_progress:
            range_mmse = tqdm(range_mmse)

        for i in range_mmse:
            end = min(i + batch_size, mmse_count) - i
            x_normalized = model.normalize_input(inp[:end])
            tar_normalized = model.normalize_target(tar[:end])
            recon_normalized, td_data = model(x_normalized)
            recon_img = model.likelihood.get_mean_lv(recon_normalized)[0]
            if padN > 0:
                tar_normalized = tar_normalized[:, :, padN:-padN, padN:-padN]
                recon_normalized = recon_normalized[:, :, padN:-padN, padN:-padN]
                recon_img = recon_img[:, :, padN:-padN, padN:-padN]

            assert tar_normalized.shape[-1] == prediction_size
            assert tar_normalized.shape[-2] == prediction_size
            assert tar_normalized.shape[-2:] == recon_normalized.shape[-2:]
            recon_img_list.append(recon_img.cpu())
        mmse_img = torch.mean(torch.cat(recon_img_list, dim=0), dim=0)[None]

    dset.set_img_sz(old_img_sz)
    return mmse_img, tar_normalized.cpu()


def get_dset_predictions(model, dset, batch_size, model_type=None, mmse_count=1, num_workers=4):
    dloader = DataLoader(dset, pin_memory=False, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    predictions = []
    predictions_std = []
    losses = []
    logvar_arr = []
    patch_psnr_channels = [RunningPSNR() for _ in range(dset[0][1].shape[0])]
    with torch.no_grad():
        for batch in tqdm(dloader):
            inp, tar = batch[:2]
            inp = inp.cuda()
            tar = tar.cuda()

            recon_img_list = []
            for mmse_idx in range(mmse_count):
                if model_type in [ModelType.UNet, ModelType.BraveNet]:
                    x_normalized = model.normalize_input(inp)
                    tar_normalized = model.normalize_target(tar)

                    recon_normalized = model(x_normalized)
                    if model_type == ModelType.BraveNet:
                        recon_normalized = recon_normalized[0]

                    imgs = recon_normalized
                    rec_loss = model.get_reconstruction_loss(recon_normalized, tar_normalized)

                    if mmse_idx == 0:
                        logvar_arr.append(np.array([-1]))
                        losses.append(rec_loss.cpu().numpy())

                else:
                    if model_type == ModelType.LadderVaeStitch:
                        x_normalized = model.normalize_input(inp)
                        tar_normalized = model.normalize_target(tar)

                        recon_normalized, td_data = model(x_normalized)
                        offset = model.compute_offset(td_data['z'])
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       tar_normalized,
                                                                       offset,
                                                                       return_predicted_img=True)
                    elif model_type == ModelType.LadderVaeSemiSupervised:
                        x_normalized = model.normalize_input(inp, torch.zeros_like(tar[:, 0, 0, 0], dtype=torch.int64))
                        tar_normalized = model.normalize_target(tar, torch.zeros_like(tar[:, 0, 0, 0],
                                                                                      dtype=torch.int64))

                        recon_normalized, td_data = model(x_normalized)
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       x_normalized,
                                                                       tar_normalized,
                                                                       return_predicted_img=True)

                    elif model_type == ModelType.LadderVaeMixedRecons:
                        x_normalized = model.normalize_input(inp)
                        tar_normalized = model.normalize_target(tar)

                        recon_normalized, td_data = model(x_normalized)
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       x_normalized,
                                                                       tar_normalized,
                                                                       return_predicted_img=True)
                    elif model_type in [
                            ModelType.LadderVaeTwoDataSet, ModelType.LadderVaeTwoDatasetMultiBranch,
                            ModelType.LadderVaeTwoDatasetMultiOptim
                    ]:
                        dset_idx, loss_idx = batch[2:]
                        dset_idx = dset_idx.cuda()
                        loss_idx = loss_idx.cuda()

                        x_normalized = model.normalize_input(inp)
                        tar_normalized = model.normalize_target(tar, dset_idx)
                        if model_type in [
                                ModelType.LadderVaeTwoDatasetMultiBranch, ModelType.LadderVaeTwoDatasetMultiOptim
                        ]:
                            mask_mixrecons = loss_idx == LossType.ElboMixedReconstruction
                            mask_2ch = loss_idx == LossType.Elbo
                            assert mask_2ch.sum() in [0, len(x_normalized)]
                            assert mask_mixrecons.sum() in [0, len(x_normalized)]
                            loss_idx_type = LossType.Elbo if mask_2ch.sum() == len(
                                x_normalized) else LossType.ElboMixedReconstruction
                            recon_normalized, _ = model(x_normalized, loss_idx_type)
                        else:
                            recon_normalized, _ = model(x_normalized)
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       tar_normalized,
                                                                       dset_idx,
                                                                       loss_idx,
                                                                       return_predicted_img=True)

                    elif model_type == ModelType.LVaeDeepEncoderIntensityAug:
                        x_normalized = model.normalize_input(inp)
                        alpha = torch.Tensor([0.5] * len(x_normalized)).to(x_normalized.device)
                        tar_normalized = model.normalize_target(tar, batch=(None, None, alpha))
                        out_l1, out_l2, td_data = model(x_normalized)

                        rec_loss, imgs = model.get_reconstruction_loss(out_l1,
                                                                       out_l2,
                                                                       tar_normalized,
                                                                       return_predicted_img=True)
                        imgs = torch.cat(imgs, dim=1)
                        rec_loss = {'loss': rec_loss}
                    elif model_type == ModelType.Denoiser:
                        assert model.denoise_channel in [
                            'Ch1', 'Ch2', 'input'
                        ], '"all" denoise channel not supported for evaluation. Pick one of "Ch1", "Ch2", "input"'

                        x_normalized_new, tar_new = model.get_new_input_target((inp, tar, *batch[2:]))
                        tar_normalized = model.normalize_target(tar_new)
                        recon_normalized, _ = model(x_normalized_new)
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       tar_normalized,
                                                                       x_normalized_new,
                                                                       return_predicted_img=True)
                    elif model_type == ModelType.DenoiserSplitter:
                        x_normalized, tar_normalized = model.get_normalized_input_target((inp, tar, *batch[2:]))
                        recon_normalized, _ = model(x_normalized)
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       tar_normalized,
                                                                       x_normalized,
                                                                       return_predicted_img=True)

                    else:
                        x_normalized = model.normalize_input(inp)
                        tar_normalized = model.normalize_target(tar)
                        recon_normalized, _ = model(x_normalized)
                        rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                                       tar_normalized,
                                                                       inp,
                                                                       return_predicted_img=True)

                    if mmse_idx == 0:
                        q_dic = model.likelihood.distr_params(recon_normalized) if model.likelihood is not None else {
                            'logvar': None
                        }
                        if q_dic['logvar'] is not None:
                            logvar_arr.append(q_dic['logvar'].cpu().numpy())
                        else:
                            logvar_arr.append(np.array([-1]))

                        try:
                            losses.append(rec_loss['loss'].cpu().numpy())
                        except:
                            losses.append(rec_loss['loss'])

                for i in range(imgs.shape[1]):
                    patch_psnr_channels[i].update(imgs[:, i], tar_normalized[:, i])

                recon_img_list.append(imgs.cpu()[None])

            samples = torch.cat(recon_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)
            mmse_std = torch.std(samples, dim=0)
            predictions.append(mmse_imgs.cpu().numpy())
            predictions_std.append(mmse_std.cpu().numpy())

    psnr = [x.get() for x in patch_psnr_channels]
    return np.concatenate(predictions,
                          axis=0), np.array(losses), np.concatenate(logvar_arr), psnr, np.concatenate(predictions_std,
                                                                                                      axis=0)
