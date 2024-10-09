import math
from typing import Union

import numpy as np
import torch
from torch import nn

from core.stable_dist_params import StableLogVar


class LikelihoodModule(nn.Module):

    def distr_params(self, x):
        return None

    def set_params_to_same_device_as(self, correct_device_tensor):
        pass

    @staticmethod
    def logvar(params):
        return None

    @staticmethod
    def mean(params):
        return None

    @staticmethod
    def mode(params):
        return None

    @staticmethod
    def sample(params):
        return None

    def log_likelihood(self, x, params):
        return None

    def forward(self, input_, x):
        distr_params = self.distr_params(input_)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        logvar = self.logvar(distr_params)
        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)
        dct = {
            'mean': mean,
            'mode': mode,
            'sample': sample,
            'params': distr_params,
            'logvar': logvar,
        }
        return ll, dct


class NoiseModelLikelihood(LikelihoodModule):

    def __init__(self, ch_in, color_channels, data_mean, data_std, noiseModel):
        super().__init__()
        self.parameter_net = nn.Identity()  #nn.Conv2d(ch_in, color_channels, kernel_size=3, padding=1)
        self.data_mean = data_mean
        self.data_std = data_std
        self.noiseModel = noiseModel

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)
        elif isinstance(self.data_mean, dict):
            for key in self.data_mean.keys():
                self.data_mean[key] = self.data_mean[key].to(correct_device_tensor.device)
                self.data_std[key] = self.data_std[key].to(correct_device_tensor.device)

    def get_mean_lv(self, x):
        return self.parameter_net(x), None

    def distr_params(self, x):
        mean, lv = self.get_mean_lv(x)
        # mean, lv = x.chunk(2, dim=1)

        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params['mean']

    def log_likelihood(self, x, params):
        predicted_s_denormalized = params['mean'] * self.data_std['target'] + self.data_mean['target']
        x_denormalized = x * self.data_std['target'] + self.data_mean['target']
        # predicted_s_cloned = predicted_s_denormalized
        # predicted_s_reduced = predicted_s_cloned.permute(1, 0, 2, 3)

        # x_cloned = x_denormalized
        # x_cloned = x_cloned.permute(1, 0, 2, 3)
        # x_reduced = x_cloned[0, ...]
        # import pdb;pdb.set_trace()
        likelihoods = self.noiseModel.likelihood(x_denormalized, predicted_s_denormalized)
        # likelihoods = self.noiseModel.likelihood(x, params['mean'])
        logprob = torch.log(likelihoods)
        return logprob


class GaussianLikelihood(LikelihoodModule):

    def __init__(self,
                 ch_in,
                 color_channels,
                 predict_logvar: Union[None, str] = None,
                 logvar_lowerbound=None,
                 conv2d_bias=True):
        super().__init__()
        # If True, then we also predict pixelwise logvar.
        self.predict_logvar = predict_logvar
        self.logvar_lowerbound = logvar_lowerbound
        self.conv2d_bias = conv2d_bias
        assert self.predict_logvar in [None, 'global', 'pixelwise', 'channelwise']
        logvar_ch_needed = self.predict_logvar is not None
        # self.parameter_net = nn.Conv2d(ch_in,
        #                                color_channels * (1 + logvar_ch_needed),
        #                                kernel_size=3,
        #                                padding=1,
        #                                bias=self.conv2d_bias)
        self.parameter_net = nn.Identity()
        print(f'[{self.__class__.__name__}] PredLVar:{self.predict_logvar} LowBLVar:{self.logvar_lowerbound}')

    def get_mean_lv(self, x):
        x = self.parameter_net(x)
        if self.predict_logvar is not None:
            # pixelwise mean and logvar
            mean, lv = x.chunk(2, dim=1)
            if self.predict_logvar in ['channelwise', 'global']:
                if self.predict_logvar == 'channelwise':
                    # logvar should be of the following shape (batch,num_channels). Other dims would be singletons.
                    N = np.prod(lv.shape[:2])
                    new_shape = (*mean.shape[:2], *([1] * len(mean.shape[2:])))
                elif self.predict_logvar == 'global':
                    # logvar should be of the following shape (batch). Other dims would be singletons.
                    N = lv.shape[0]
                    new_shape = (*mean.shape[:1], *([1] * len(mean.shape[1:])))
                else:
                    raise ValueError(f"Invalid value for self.predict_logvar:{self.predict_logvar}")

                lv = torch.mean(lv.reshape(N, -1), dim=1)
                lv = lv.reshape(new_shape)

            if self.logvar_lowerbound is not None:
                lv = torch.clip(lv, min=self.logvar_lowerbound)
        else:
            mean = x
            lv = None
        return mean, lv

    def distr_params(self, x):
        mean, lv = self.get_mean_lv(x)

        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params['mean']

    @staticmethod
    def logvar(params):
        return params['logvar']

    def log_likelihood(self, x, params):
        if self.predict_logvar is not None:
            logprob = log_normal(x, params['mean'], params['logvar'])
        else:
            logprob = -0.5 * (params['mean'] - x)**2
        return logprob


def log_normal(x, mean, logvar):
    """
    Log of the probability density of the values x untder the Normal
    distribution with parameters mean and logvar.
    :param x: tensor of points, with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param logvar: tensor with log-variance of distribution, shape has to be
                   either scalar or broadcastable
    """
    var = torch.exp(logvar)
    log_prob = -0.5 * (((x - mean)**2) / var + logvar + torch.tensor(2 * math.pi).log())
    return log_prob


class GaussianLikelihoodWithStitching(GaussianLikelihood):

    def forward(self, input_, x, offset):
        distr_params = self.distr_params(input_)
        distr_params['mean'] = distr_params['mean'] + offset

        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        logvar = self.logvar(distr_params)
        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)
        dct = {
            'mean': mean,
            'mode': mode,
            'sample': sample,
            'params': distr_params,
            'logvar': logvar,
        }
        return ll, dct
