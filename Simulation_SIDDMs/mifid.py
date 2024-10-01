import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import math
import os
import glob
from tqdm import tqdm
from PIL import Image
from scipy import linalg

def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_FID_infinity_(real, fake, batch_size=50, min_fake=5000, num_points=15):
    """
    Calculates effectively unbiased FID_inf using extrapolation given
    paths to real and fake data
    Args:
        real_path: (str)
            Path to real dataset or precomputed .npz statistics.
        fake_path: (str)
            Path to fake dataset.
        batch_size: (int)
            The batch size for dataloader.
            Default: 50
        min_fake: (int)
            Minimum number of images to evaluate FID on.
            Default: 5000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    """
    # load pretrained inception model

    # get all activations of generated images
    real_m, real_s = real

    fake_act = fake

    num_fake = len(fake_act)
    assert num_fake > min_fake, \
        'number of fake data must be greater than the minimum point for extrapolation'

    fids = []

    # Choose the number of images to evaluate FID_N at regular intervals over N
    fid_batches = np.linspace(min_fake, num_fake, num_points).astype('int32')

    # Evaluate FID_N
    for fid_batch_size in fid_batches:
        # sample with replacement
        np.random.shuffle(fake_act)
        fid_activations = fake_act[:fid_batch_size]
        m, s = np.mean(fid_activations, axis=0), np.cov(fid_activations, rowvar=False)
        FID = numpy_calculate_frechet_distance(m, s, real_m, real_s)
        fids.append(FID)
    fids = np.array(fids).reshape(-1, 1)

    # Fit linear regression
    reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0, 0]

    return fid_infinity