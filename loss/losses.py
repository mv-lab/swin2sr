import datetime
import os
import time
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

from utils import utils


def free_bits_kl(kl, free_bits, batch_average=False, eps=1e-6) -> torch.Tensor:
    """Computes free-bits version of KL divergence.
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).
    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)
    Returns:
        The KL with free bits
    """

    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)


def lossFunctionKLD(mu, logvar):
    """Compute KL divergence loss.
    Parameters
    ----------
    mu: Tensor
        Latent space mean of encoder distribution.
    logvar: Tensor
        Latent space log variance of encoder distribution.
    """
    kl_error = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_error


def recoLossGaussian(predicted_x, x, gaussian_noise_std, data_std):
    """
    Compute reconstruction loss for a Gaussian noise model.
    This is essentially the MSE loss with a factor depending on the standard deviation.
    Parameters
    ----------
    predicted_x: Tensor
        Predicted signal by disentangle decoder.
    x: Tensor
        Noisy observation image.
    gaussian_noise_std: float
        Standard deviation of Gaussian noise.
    data_std: float
        Standard deviation of training and validation data combined (used for normailzation).
    """
    reconstruction_error = torch.mean((predicted_x - x)**2) / (2.0 * (gaussian_noise_std / data_std)**2)
    return reconstruction_error


def recoLoss(predicted_x, x, data_mean, data_std, noiseModel):
    """Compute reconstruction loss for an arbitrary noise model.
    Parameters
    ----------
    predicted_x: Tensor
        Predicted signal by disentangle decoder.
    x: Tensor
        Noisy observation image.
    data_mean: float
        Mean of training and validation data combined (used for normailzation).
    data_std: float
        Standard deviation of training and validation data combined (used for normailzation).
    device: GPU device
        torch cuda device
    """
    predicted_x_denormalized = predicted_x * data_std + data_mean
    x_denormalized = x * data_std + data_mean
    predicted_x_cloned = predicted_x_denormalized
    predicted_x_reduced = predicted_x_cloned.permute(1, 0, 2, 3)

    x_cloned = x_denormalized
    x_cloned = x_cloned.permute(1, 0, 2, 3)
    x_reduced = x_cloned[0, ...]

    likelihoods = noiseModel.likelihood(x_reduced, predicted_x_reduced)
    log_likelihoods = torch.log(likelihoods)

    # Sum over pixels and batch
    reconstruction_error = -torch.mean(log_likelihoods)
    return reconstruction_error


def vanilla_vae_loss_fn(predicted_x, x, mu, logvar):
    """Compute VAE elbo loss.
    Parameters
    ----------
    predicted_x: Tensor
        Predicted signal by disentangle decoder.
    x: Tensor
        Noisy observation image.
    mu: Tensor
        Latent space mean of encoder distribution.
    logvar: Tensor
        Latent space logvar of encoder distribution.
    """
    kl_loss = lossFunctionKLD(mu, logvar)
    reconstruction_loss = recoLossGaussian(predicted_x, x, 1, 1)
    return kl_loss / float(x.numel()), reconstruction_loss


def disentangle_loss_fn(predicted_x, x, mu, logvar, gaussian_noise_std, data_mean, data_std, noiseModel):
    """Compute disentangle loss.
    Parameters
    ----------
    predicted_x: Tensor
        Predicted signal by disentangle decoder.
    x: Tensor
        Noisy observation image.
    mu: Tensor
        Latent space mean of encoder distribution.
    logvar: Tensor
        Latent space logvar of encoder distribution.
    gaussian_noise_std: float
        Standard deviation of Gaussian noise (required when using Gaussian reconstruction loss).
    data_mean: float
        Mean of training and validation data combined (used for normailzation).
    data_std: float
        Standard deviation of training and validation data combined (used for normailzation).
    device: GPU device
        torch cuda device
    noiseModel: NoiseModel object
        Distribution of noisy pixel values corresponding to clean signal (required when using general reconstruction loss).
    """
    kl_loss = lossFunctionKLD(mu, logvar)

    if noiseModel is not None:
        reconstruction_loss = recoLoss(predicted_x, x, data_mean, data_std, noiseModel)
    else:
        reconstruction_loss = recoLossGaussian(predicted_x, x, gaussian_noise_std, data_std)
    #print(float(x.numel()))
    return reconstruction_loss, kl_loss / float(x.numel())


class Elbo(nn.Module):
    def forward(self, predicted_x, x, mu, logvar):
        kl, recons = vanilla_vae_loss_fn(predicted_x, x, mu, logvar)
        return {'kl': kl, 'recons': recons}