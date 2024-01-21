
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmd_metric import polynomial_mmd, calculate_frechet_distance
import argparse
import time
import seaborn as sns
import random
from diffusion_utils import *


sns.set(color_codes=True)

import numpy as np

def sample_from_model(generator, n_time, x_init):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t_p1 = torch.full((x.size(0),), i + 1, dtype=torch.float).to(x.device) / n_time
            x_0 = generator(x, t_p1, torch.rand_like(x))
            if i > 0:
                t = torch.full((x.size(0),), i, dtype=torch.float).to(x.device) / n_time
                x = get_z_t_via_z_tp1(x_0, x, t, t_p1)#get_z_t(x_0, t)#
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
#             t = torch.full((x.size(0),), i, dtype=torch.float).to(x.device) / n_time#t_p1 - 1/train_n_time
#             latent_z = torch.randn(x.size(0), nz, device=x.device)
#             x_0 = generator(x, t_p1, latent_z)
#             if float(i+1) / float(n_time) - 1.0/float(train_n_time) > 0:
#                 x = get_z_t_via_z_tp1(x_0, x, t, t_p1)
#                 # t_p1 = torch.full((x.size(0),), i, dtype=torch.float).to(x.device) / n_time
#                 # x, _, _ = get_z_tp1_via_z_t(t, t_p1, x)
#                 x = x.detach()
#             else:
#                 x = x_0.detach()
#                 break
#     return x

def plot_y_givenx(x,y,distirbution):
    link = x.mean(dim=1)
    label = y[np.logical_and(link.numpy() >= 2.9, link.numpy() <= 3.1)]
    fig, ax = plt.subplots(1, 1)
    if distirbution=='poisson':
        a_plot = sns.distplot(label.numpy())
    else:
        a_plot = sns.kdeplot(label.numpy(), shade=True, shade_lowest=False)
    plt.title('Conditional Y|X')
    label = np.expand_dims(label,axis=1)
    return fig, label

def get_y(data,distribution=None):
    label = []
    if distribution == 'poisson':
        for x in data:
            y = np.random.poisson(x.mean())
            label.append(y)
    elif distribution == 'normal':
        for x in data:
            y = np.random.normal(x.mean())
            label.append(y)
    elif distribution == 'gamma':
        for x in data:
            y = np.random.gamma(x.mean())
            label.append(y)
    elif distribution == 'inv_gauss':
        for x in data:
            y = np.random.wald(1/np.sqrt(x.mean()),1)
            label.append(y)

    label = np.asarray(label)
    return label

def glm_loss(link,y,distribution=None):
    if distribution == 'poisson':
        loss = -(y*link - torch.exp(link)).mean()
    elif distribution == 'normal':
        loss = ((y-link)*(y-link)).mean()
    elif distribution == 'gamma':
        link = link.clamp(min=1e-10)
        loss = -(-y*link-torch.log(1/link)).mean()
    elif distribution == 'inv_gauss':
        link = link.clamp(min= 1e-10)
        loss = (link*y/2 - torch.sqrt(link)).mean()
            #((y-1/torch.sqrt(link))*(y-1/torch.sqrt(link))/(2*y/link)).mean()
    return loss*10


def plot_density(flights,binwidth=0.1):
    ax = plt.subplot(1,1,1)

    # Draw the plot
    ax.hist(flights, bins=int(180 / binwidth),
            color='blue', edgecolor='black')

    # Title and labels
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size=30)
    ax.set_xlabel('Delay (min)', size=22)
    ax.set_ylabel('Flights', size=22)


    plt.tight_layout()


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class G_guassian(nn.Module):

    def __init__(self, nz, t_emb_dim=128, act=nn.LeakyReLU(0.2), ch=100):
        super(G_guassian, self).__init__()

        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.decode = nn.Sequential(

            nn.Linear(nz*2+t_emb_dim,ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, 2),
        )
        self.out = nn.Linear(4,2)


        self.__initialize_weights()

    def forward(self, xt, t, z):

        t_embed = self.act(self.t_embed(t*1000.0))
        input = torch.cat([xt, t_embed, z], dim=1)
        output = torch.cat([self.decode(input),xt], dim=1)
        output = self.out(output)

        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class D_guassian(nn.Module):

    def __init__(self, args, t_emb_dim=128, act=nn.LeakyReLU(0.2), ch=100):
        super(D_guassian, self).__init__()

        if args.ac_w>0:
            self.joint = 0.0
        else:
            self.joint = 1.0

        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.encode = nn.Sequential(

            nn.Linear(2*2+t_emb_dim, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.SiLU(),
        )
        self.gan_linear = nn.Linear(ch, 2)
        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input, t, input_tp1):

        t_embed = self.act(self.t_embed(t*1000.0))

        input = torch.cat([input, t_embed, input_tp1*self.joint], dim=1)

        x = self.encode(input)

        s = self.gan_linear(x)
        return self.sigmoid(s).squeeze()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

def multi_results(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    nz = 2

    G = G_guassian(nz=nz).cuda()
    D = D_guassian(args=args).cuda()


    optg = optim.Adam(G.parameters(), lr=1e-4,
                      betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=4e-4,
                      betas=(0.5, 0.999))

    save_path = os.path.join('./MOG/siddms/', '2D_' + str(args.num_timesteps) +
                             'xt_continuous_ddgan_x_0gan_' + 'AC_w' + str(args.ac_w))
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)

    data = []
    for i in range(5):
        for j in range(5):
            random_data = (torch.randn(10000, 2).cuda()/3.0).clamp(min=-1, max=1)
            random_data[:,0] = random_data[:,0]+ i*5
            random_data[:, 1] = random_data[:, 1]+ j * 5
            data.append(random_data)

    data = torch.cat(data, dim=0).cuda()
    fig, ax = plt.subplots(1, 1)
    plt.axis('equal')
    # plt.xlim([0.0,6.0])
    # plt.ylim([0.0,6.0])
    sns.kdeplot(data.cpu().numpy()[:,0], data.cpu().numpy()[:, 1],cmap="Blues", shade=True, shade_lowest=False,cbar=True)
    plt.title('Marginal')
    fig.savefig(save_path + '/o_marginal.png')
    fig.show()

    r_data = data.cpu().numpy()
    np.save(save_path + '/o_data', r_data)


    #diffusion gan
    for j in range(300):
        for i in range(625):

            # D step
            for _ in range(1):

                index = np.random.choice(250000, 4000, replace=False)
                train_data = data[index, ...]

                t_p1 = (torch.rand(train_data.shape[0])).to(train_data.device).clamp(
                    min=1e-4, max=1.0)
                t = (t_p1 - 1.0 / args.num_timesteps).clamp(min=0.0)

                x_t, _ = get_z_t(train_data, t)
                x_tp1 = get_z_tp1_via_z_t(t, t_p1, x_t)

                #D real
                d_real = D(train_data, t, x_tp1)

                #D fake

                x_0_predict = G(x_tp1.detach(), t_p1, torch.rand_like(x_tp1))
                # x_pos_sample, _ = get_z_t(x_0_predict, t)
                x_pos_sample = get_z_t_via_z_tp1(x_0_predict, x_tp1, t, t_p1)
                d_fake = D(x_0_predict, t, x_tp1)

                D_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + \
                         F.binary_cross_entropy(d_fake,torch.zeros_like(d_fake))
                optd.zero_grad()
                D_loss.backward()
                optd.step()

            # G step
            if i % 1 == 0:

                t_p1 = (torch.rand(train_data.shape[0])).to(train_data.device).clamp(
                    min=1e-4, max=1.0)
                t = (t_p1 - 1.0 / args.num_timesteps).clamp(min=0.0)

                x_tp1, _ = get_z_t(train_data, t_p1)

                x_0_predict = G(x_tp1, t_p1, torch.rand_like(x_tp1))
                x_pos_sample = get_z_t_via_z_tp1(x_0_predict, x_tp1, t, t_p1)
                # x_tp1_mean_fake, sigma = get_z_t_(x_0_predict, t_p1)
                x_tp1_fake = get_z_tp1_via_z_t(t, t_p1, x_pos_sample)

                x_tp1_mean_real, _ = get_z_t_(train_data, t_p1)

                g_fake = D(x_0_predict, t, x_tp1)

                G_loss = F.binary_cross_entropy(g_fake, torch.ones_like(g_fake))

                error_condition = torch.square(x_tp1_fake - x_tp1).mean()
                # error_condition = torch.square(get_mu_posterior(t, t_p1, train_data) - get_mu_posterior(t, t_p1, x_0_predict)).mean()

                optg.zero_grad()
                (G_loss+error_condition*args.ac_w).backward()
                optg.step()

        print(j, D_loss.item(), G_loss.item(), error_condition.item())

    torch.save(G.state_dict(), os.path.join(save_path, 'netG.pth'))
    # G.load_state_dict(torch.load(os.path.join(save_path, 'netG.pth')))
    results = []
    for num_step in range(args.num_timesteps):
        num_step = num_step + 1
        with torch.no_grad():

            data_gs = []
            for _ in range(10):
                x_t_1 = torch.randn(50000, nz).cuda()
                data_g = sample_from_model(G, num_step, x_t_1)
                data_gs.append(data_g.cpu())

            data_gs = torch.cat(data_gs, dim=0)

            fig, ax = plt.subplots(1, 1)

            ax.set_xlim(xmin=-2.5, xmax=22.5)
            ax.set_ylim(ymin=-2.5, ymax=22.5)
            ax.set_aspect('equal', 'box')

            sns.kdeplot(data_gs.numpy()[:, 0], data_gs.numpy()[:, 1], cmap="Blues", shade=True, shade_lowest=False,
                        cbar=False)
            plt.title('T=' + str(args.num_timesteps), fontsize=20)
            fig.savefig(save_path + '/marginal_'+str(num_step)+'.png')

            g_data = data_gs.cpu().numpy()

            np.save(save_path + '/g_data_'+str(num_step), g_data)

            mu = np.mean(r_data, axis=0)
            sigma = np.cov(r_data, rowvar=False)

            mu_g = np.mean(g_data, axis=0)
            sigma_g = np.cov(g_data, rowvar=False)

            fid = calculate_frechet_distance(mu, sigma, mu_g, sigma_g)


            data_r = data.cpu().numpy()
            mu = np.mean(data_r)
            std = np.std(data_r)

            data_r = (data_r - mu) / std
            data_g = (g_data - mu) / std
            mean0_0, var0_0 = polynomial_mmd(data_g, data_r)
            print(fid)
            print(mean0_0, var0_0)

            #
            results.append(str(fid))
            results.append(str(mean0_0))
            results.append(str(var0_0))
            #

        file = open(save_path + '/result.text', 'w')

        for content in results:
            file.write(content + '\n')






if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--ac_w', type=float, default=0.5,
                        help='AC weight')
    parser.add_argument('--use_AC', action='store_true', default=False)


    args = parser.parse_args()

    multi_results(args)