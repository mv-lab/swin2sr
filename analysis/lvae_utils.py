import numpy as np
import torch

from core.data_utils import crop_img_tensor


def get_img_from_forward_output(out, model):
    recons_img = model.likelihood.get_mean_lv(out)[0]
    recons_img = recons_img * model.data_std + model.data_mean
    return recons_img


def get_z(img, model):
    with torch.no_grad():
        img = torch.Tensor(img[None]).cuda()
        x_normalized = model.normalize(img)
        recons_img_latent, td_data = model(x_normalized)
        q_mu = td_data['q_mu']
        recons_img = get_img_from_forward_output(recons_img_latent, model)
        return recons_img, q_mu


def get_recons_with_latent(img_shape, z, model):
    # Top-down inference/generation
    out, td_data = model.topdown_pass(None, forced_latent=z, n_img_prior=1)
    # Restore original image size
    out = crop_img_tensor(out, img_shape)

    return get_img_from_forward_output(out, model)
