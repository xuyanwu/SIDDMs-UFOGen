import numpy as np
import torch
from torch import nn
from mmd_metric import polynomial_mmd, calculate_frechet_distance


data_g = np.load("MOG/2D_4xt_continuous_forward_AC_w1.0/g_data_4.npy")
data_r = np.load("MOG/2D_4xt_continuous_forward_AC_w1.0/o_data.npy")

mu = np.mean(data_r)
std = np.std(data_r)

data_r = (data_r-mu)/std
data_g = (data_g-mu)/std

mean, var = polynomial_mmd(data_g, data_r)

print(mean, var)