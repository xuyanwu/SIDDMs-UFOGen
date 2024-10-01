
import torch
import numpy as np
import math
import torch.nn.functional as F

# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):

        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)

def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    mu = extract(coeff.a_s, t + 1, x_start.shape) * x_t
    sigma = extract(coeff.sigmas, t + 1, x_start.shape)
    x_t_plus_one = mu + sigma * noise

    return x_t, x_t_plus_one, mu, sigma

def q_sample_xtp1(coeff, x_t, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_t)
    mu = extract(coeff.a_s, t + 1, x_t.shape) * x_t
    sigma = extract(coeff.sigmas, t + 1, x_t.shape)
    x_t_plus_one = mu + sigma * noise

    return x_t_plus_one, mu


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):

        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32 ,device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = \
                    ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos

def sample_from_model(generator, n_time, x_init, nz):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t_p1 = torch.full((x.size(0),), i + 1, dtype=torch.float).to(x.device) / n_time
            t = torch.full((x.size(0),), i, dtype=torch.float).to(x.device) / n_time
            latent_z = torch.randn(x.size(0), nz, device=x.device)
            x_0 = generator(x, t_p1, latent_z)
            if i > 0:
                x = get_z_t_via_z_tp1(x_0, x, t, t_p1)
                x_new = x.detach()
            else:
                x_new = x_0.detach()
                break
    return x_new

# def sample_from_model(generator, n_time, train_n_time, x_init, nz):
#     x = x_init
#     with torch.no_grad():
#         for i in reversed(range(n_time)):
#             t_p1 = torch.full((x.size(0),), i + 1, dtype=torch.float).to(x.device) / n_time
#             t = t_p1 - 1/train_n_time
#             latent_z = torch.randn(x.size(0), nz, device=x.device)
#             x_0 = generator(x, t_p1, latent_z)
#             if (i + 1) / n_time - 1/train_n_time > 0:
#                 x = get_z_t_via_z_tp1(x_0, x, t, t_p1)
#                 t_p1 = torch.full((x.size(0),), i, dtype=torch.float).to(x.device) / n_time
#                 x, _, _ = get_z_tp1_via_z_t(t, t_p1, x)
#                 x = x.detach()
#             else:
#                 x = x_0.detach()
#                 break
#     return x

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

def get_alpha_cum(t):
    return (torch.cos((t + 0.008) / 1.008 * math.pi / 2).clamp(min=0.0, max=1.0))**2

# def get_alpha_cum(t):
#     return (1-t).clamp(min=0.0, max=1.0)


def get_z_t(x_0, t, eps=None):
    alpha_cum = get_alpha_cum(t)[:,None]
    if eps is None:
        eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*eps
    return x_t

def get_eps_x_t(x_0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:,None]
    eps = (x_t - torch.sqrt(alpha_cum)*x_0)/torch.sqrt(1-alpha_cum)
    return eps

def get_z_t_(x_0, t):
    alpha_cum = get_alpha_cum(t)[:,None]
    return torch.sqrt(alpha_cum)*x_0

def get_z_tp1_via_z_t(t, t_p1, z_t):
    alpha_cum = get_alpha_cum(t)[:, None]
    alpha_cum_p1 = get_alpha_cum(t_p1)[:, None]
    beta_p1 = 1 - alpha_cum_p1/alpha_cum
    return torch.sqrt(1-beta_p1)*z_t + torch.sqrt(beta_p1)*torch.randn_like(z_t)

def get_mu_posterior(t, t_p1, x_0):
    alpha_cum = get_alpha_cum(t)[:, None]
    alpha_cum_p1 = get_alpha_cum(t_p1)[:, None]
    beta_p1 = 1 - alpha_cum_p1/alpha_cum
    mu = torch.sqrt(alpha_cum_p1)*beta_p1/(1-alpha_cum_p1)*x_0
    return mu

def get_z_t_via_z_tp1(x_0, z_tp1, t, t_p1):
    alpha_cum = get_alpha_cum(t)[:, None]
    alpha_cum_p1 = get_alpha_cum(t_p1)[:, None]
    beta_p1 = 1 - alpha_cum_p1/alpha_cum
    mean_0 = torch.sqrt(alpha_cum)*beta_p1/(1-alpha_cum_p1)
    mean_tp1 = torch.sqrt(1-beta_p1)*(1-alpha_cum)/(1-alpha_cum_p1)

    var = (1-alpha_cum)/(1-alpha_cum_p1)*beta_p1

    return mean_0*x_0 + mean_tp1*z_tp1 + torch.sqrt(var)*torch.randn_like(x_0)


def ddim_sample(x_0, z_tp1, t, t_p1):
    epsilon = get_eps_x_t(x_0, z_tp1, t_p1)
    alpha_cum = get_alpha_cum(t)[:, None]
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*epsilon
    return x_t

