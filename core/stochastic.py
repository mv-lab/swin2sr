"""
Adapted from https://github.com/juglab/HDN/blob/e30edf7ec2cd55c902e469b890d8fe44d15cbb7e/lib/stochastic.py
"""
import math
from typing import Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from denoisplit.core.stable_dist_params import StableLogVar, StableMean
from denoisplit.core.stable_exp import log_prob


class NormalStochasticBlock2d(nn.Module):
    """
    Transform input parameters to q(z) with a convolution, optionally do the
    same for p(z), then sample z ~ q(z) and return conv(z).
    If q's parameters are not given, do the same but sample from p(z).
    """

    def __init__(self,
                 c_in: int,
                 c_vars: int,
                 c_out,
                 kernel: int = 3,
                 transform_p_params: bool = True,
                 use_naive_exponential=False):
        """
        Args:
            c_in:   This is the channel count of the tensor input to this module.
            c_vars: This is the size of the latent space
            c_out:  Output of the stochastic layer. Note that this is different from z.
            kernel: kernel used in convolutional layers.
            transform_p_params: p_params are transformed if this is set to True.
        """
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self._use_naive_exponential = use_naive_exponential

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = nn.Conv2d(c_vars, c_out, kernel, padding=pad)

    # def forward_swapped(self, p_params, q_mu, q_lv):
    #
    #     if self.transform_p_params:
    #         p_params = self.conv_in_p(p_params)
    #     else:
    #         assert p_params.size(1) == 2 * self.c_vars
    #
    #     # Define p(z)
    #     p_mu, p_lv = p_params.chunk(2, dim=1)
    #     p = Normal(p_mu, (p_lv / 2).exp())
    #
    #     # Define q(z)
    #     q = Normal(q_mu, (q_lv / 2).exp())
    #     # Sample from q(z)
    #     sampling_distrib = q
    #
    #     # Generate latent variable (typically by sampling)
    #     z = sampling_distrib.rsample()
    #
    #     # Output of stochastic layer
    #     out = self.conv_out(z)
    #
    #     data = {
    #         'z': z,  # sampled variable at this layer (batch, ch, h, w)
    #         'p_params': p_params,  # (b, ch, h, w) where b is 1 or batch size
    #     }
    #     return out, data

    def get_z(self, sampling_distrib, forced_latent, use_mode, mode_pred, use_uncond_mode):

        # Generate latent variable (typically by sampling)
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                if mode_pred:
                    if use_uncond_mode:
                        z = sampling_distrib.mean

                    else:
                        z = sampling_distrib.rsample()
                else:
                    z = sampling_distrib.rsample()
        else:
            z = forced_latent
        return z

    def sample_from_q(self, q_params, var_clip_max):
        """
        Note that q_params should come from outside. It must not be already transformed since we are doing it here.
        """
        _, _, q = self.process_q_params(q_params, var_clip_max)
        return q.rsample()

    def compute_kl_metrics(self, p, p_params, q, q_params, mode_pred, analytical_kl, z):
        """
        Compute KL (analytical or MC estimate) and then process it in multiple ways.
        """
        if mode_pred is False:  # if not predicting
            if analytical_kl:
                kl_elementwise = kl_divergence(q, p)
            else:
                kl_elementwise = kl_normal_mc(z, p_params, q_params)
            kl_samplewise = kl_elementwise.sum((1, 2, 3))
            kl_channelwise = kl_elementwise.sum((2, 3))
            # Compute spatial KL analytically (but conditioned on samples from
            # previous layers)
            kl_spatial = kl_elementwise.sum(1)
        else:  # if predicting, no need to compute KL
            kl_elementwise = kl_samplewise = kl_spatial = kl_channelwise = None

        kl_dict = {
            'kl_elementwise': kl_elementwise,  # (batch, ch, h, w)
            'kl_samplewise': kl_samplewise,  # (batch, )
            'kl_spatial': kl_spatial,  # (batch, h, w)
            'kl_channelwise': kl_channelwise  # (batch, ch)
        }
        return kl_dict

    def process_p_params(self, p_params, var_clip_max):
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        if var_clip_max is not None:
            p_lv = torch.clip(p_lv, max=var_clip_max)

        p_mu = StableMean(p_mu)
        p_lv = StableLogVar(p_lv, enable_stable=not self._use_naive_exponential)
        p = Normal(p_mu.get(), p_lv.get_std())
        return p_mu, p_lv, p

    def process_q_params(self, q_params, var_clip_max, allow_oddsizes=False):
        # Define q(z)
        q_params = self.conv_in_q(q_params)
        q_mu, q_lv = q_params.chunk(2, dim=1)
        if var_clip_max is not None:
            q_lv = torch.clip(q_lv, max=var_clip_max)

        if q_mu.shape[-1] % 2 == 1 and allow_oddsizes is False:
            q_mu = F.center_crop(q_mu, q_mu.shape[-1] - 1)
            q_lv = F.center_crop(q_lv, q_lv.shape[-1] - 1)
            # clip_start = np.random.rand() > 0.5
            # q_mu = q_mu[:, :, 1:, 1:] if clip_start else q_mu[:, :, :-1, :-1]
            # q_lv = q_lv[:, :, 1:, 1:] if clip_start else q_lv[:, :, :-1, :-1]

        q_mu = StableMean(q_mu)
        q_lv = StableLogVar(q_lv, enable_stable=not self._use_naive_exponential)
        q = Normal(q_mu.get(), q_lv.get_std())
        return q_mu, q_lv, q

    def forward(self,
                p_params: torch.Tensor,
                q_params: torch.Tensor = None,
                forced_latent: Union[None, torch.Tensor] = None,
                use_mode: bool = False,
                force_constant_output: bool = False,
                analytical_kl: bool = False,
                mode_pred: bool = False,
                use_uncond_mode: bool = False,
                var_clip_max: Union[None, float] = None):
        """
        Args:
            p_params: this is passed from top layers.
            q_params: this is the merge of bottom up layer at this level and top down layers above this level.
            forced_latent: If this is a tensor, then in stochastic layer, we don't sample by using p() & q(). We simply 
                            use this as the latent space sampling.
            use_mode:   If it is true, we still don't sample from the q(). We simply 
                            use the mean of the distribution as the latent space.
            force_constant_output: This ensures that only the first sample of the batch is used. Typically used 
                                when infernce_mode is False
            analytical_kl: If True, typical KL divergence is calculated. Otherwise, a one-sample approximate of it is
                            calculated.
            mode_pred: If True, then only prediction happens. Otherwise, KL divergence loss also gets computed.
            use_uncond_mode: Used only when mode_pred=True
            var_clip_max: This is the maximum value the log of the variance of the latent vector for any layer can reach.
            
        """

        debug_qvar_max = 0
        assert (forced_latent is None) or (not use_mode)

        p_mu, p_lv, p = self.process_p_params(p_params, var_clip_max)

        p_params = (p_mu, p_lv)

        if q_params is not None:
            # At inference time, just don't centercrop the q_params even if they are odd in size.
            q_mu, q_lv, q = self.process_q_params(q_params, var_clip_max, allow_oddsizes=mode_pred is True)
            q_params = (q_mu, q_lv)
            debug_qvar_max = torch.max(q_lv.get())
            # Sample from q(z)
            sampling_distrib = q
            q_size = q_mu.get().shape[-1]
            if p_mu.get().shape[-1] != q_size and mode_pred is False:
                p_mu.centercrop_to_size(q_size)
                p_lv.centercrop_to_size(q_size)
        else:
            # Sample from p(z)
            sampling_distrib = p

        # Generate latent variable (typically by sampling)
        z = self.get_z(sampling_distrib, forced_latent, use_mode, mode_pred, use_uncond_mode)

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = (p_params[0][0:1].expand_as(p_params[0]).clone(),
                        p_params[1][0:1].expand_as(p_params[1]).clone())

        # Output of stochastic layer
        out = self.conv_out(z)

        # Compute log p(z)# NOTE: disabling its computation.
        # if mode_pred is False:
        #     logprob_p =  p.log_prob(z).sum((1, 2, 3))
        # else:
        #     logprob_p = None

        if q_params is not None:
            # Compute log q(z)
            logprob_q = q.log_prob(z).sum((1, 2, 3))
            # compute KL divergence metrics
            kl_dict = self.compute_kl_metrics(p, p_params, q, q_params, mode_pred, analytical_kl, z)
        else:
            kl_dict = {}
            logprob_q = None

        data = kl_dict
        data['z'] = z  # sampled variable at this layer (batch, ch, h, w)
        data['p_params'] = p_params  # (b, ch, h, w) where b is 1 or batch size
        data['q_params'] = q_params  # (batch, ch, h, w)
        # data['logprob_p'] = logprob_p  # (batch, )
        data['logprob_q'] = logprob_q  # (batch, )
        data['qvar_max'] = debug_qvar_max

        return out, data


def kl_normal_mc(z, p_mulv, q_mulv):
    """
    One-sample estimation of element-wise KL between two diagonal
    multivariate normal distributions. Any number of dimensions,
    broadcasting supported (be careful).
    :param z:
    :param p_mulv:
    :param q_mulv:
    :return:
    """
    assert isinstance(p_mulv, tuple)
    assert isinstance(q_mulv, tuple)
    p_mu, p_lv = p_mulv
    q_mu, q_lv = q_mulv

    p_std = p_lv.get_std()
    q_std = q_lv.get_std()

    p_distrib = Normal(p_mu.get(), p_std)
    q_distrib = Normal(q_mu.get(), q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)

    # the prior


def log_Normal_diag(x, mean, log_var):
    constant = -0.5 * torch.log(torch.Tensor([2 * math.pi])).item()
    log_normal = constant + -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    return log_normal
