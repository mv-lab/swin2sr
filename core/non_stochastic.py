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


class NonStochasticBlock2d(nn.Module):
    """
    Non-stochastic version of the NormalStochasticBlock2d
    """

    def __init__(self,
                 c_in: int,
                 c_vars: int,
                 c_out,
                 kernel: int = 3,
                 groups=1,
                 conv2d_bias: bool = True,
                 transform_p_params: bool = True):
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

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad, bias=conv2d_bias, groups=groups)
        self.conv_in_q = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad, bias=conv2d_bias, groups=groups)
        self.conv_out = nn.Conv2d(c_vars, c_out, kernel, padding=pad, bias=conv2d_bias, groups=groups)

    def compute_kl_metrics(self, p, p_params, q, q_params, mode_pred, analytical_kl, z):
        """
        Compute KL (analytical or MC estimate) and then process it in multiple ways.
        """

        kl_dict = {
            'kl_elementwise': None,  # (batch, ch, h, w)
            'kl_samplewise': None,  # (batch, )
            'kl_spatial': None,  # (batch, h, w)
            'kl_channelwise': None  # (batch, ch)
        }
        return kl_dict

    def process_p_params(self, p_params, var_clip_max):
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:

            assert p_params.size(1) == 2 * self.c_vars, f'{p_params.shape} {self.c_vars}'

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        return p_mu, None

    def process_q_params(self, q_params, var_clip_max, allow_oddsizes=False):
        # Define q(z)
        q_params = self.conv_in_q(q_params)
        q_mu, q_lv = q_params.chunk(2, dim=1)

        if q_mu.shape[-1] % 2 == 1 and allow_oddsizes is False:
            q_mu = F.center_crop(q_mu, q_mu.shape[-1] - 1)

        return q_mu, None

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

        p_mu, _ = self.process_p_params(p_params, var_clip_max)

        p_params = (p_mu, None)

        if q_params is not None:
            # At inference time, just don't centercrop the q_params even if they are odd in size.
            q_mu, _ = self.process_q_params(q_params, var_clip_max, allow_oddsizes=mode_pred is True)
            q_params = (q_mu, None)
            debug_qvar_max = torch.Tensor([1]).to(q_mu.device)
            # Sample from q(z)
            sampling_distrib = q_mu
            q_size = q_mu.shape[-1]
            if p_mu.shape[-1] != q_size and mode_pred is False:
                p_mu.centercrop_to_size(q_size)
        else:
            # Sample from p(z)
            sampling_distrib = p_mu

        # Generate latent variable (typically by sampling)
        z = sampling_distrib

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = (p_params[0][0:1].expand_as(p_params[0]).clone(),
                        p_params[1][0:1].expand_as(p_params[1]).clone())

        # Output of stochastic layer
        out = self.conv_out(z)

        kl_dict = {}
        logprob_q = None

        data = kl_dict
        data['z'] = z  # sampled variable at this layer (batch, ch, h, w)
        data['p_params'] = p_params  # (b, ch, h, w) where b is 1 or batch size
        data['q_params'] = q_params  # (batch, ch, h, w)
        data['logprob_q'] = logprob_q  # (batch, )
        data['qvar_max'] = debug_qvar_max

        return out, data
