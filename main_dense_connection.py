import os
import argparse
import math
import numpy as np
import torch
from tqdm import tqdm, trange
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plt_config

from cluster_finding_dense import find_cluster_dense_connection

matplotlib.use('Agg')
color = plt.rcParams['axes.prop_cycle'].by_key()['color']



# torch.set_default_tensor_type(torch.FloatTensor)


class Model:
    def __init__(self, batch, L, n_layers=11, Jz=1, gamma=0.2, g=2, delta=0.75, dt=0.05, init_scale=1,
                 J=None, Jz_std=0, repeat_idx=0, save=True, data_dir='/mnt/out/data'):
        self.batch = batch
        self.L = L
        self.n_layers = n_layers
        self.Jz = Jz
        self.Jz_std = Jz_std
        self.g = g
        self.gamma = gamma
        self.delta = delta
        self.dt = dt
        self.init_scale = init_scale
        self.repeat_idx = repeat_idx
        self.save = save

        # coordinates: (x, y, z)
        # memory variables on the z=0 plane
        if J is None:
            self.J = 2 * torch.randint(0, 2, (3, batch, L, L, n_layers)).float() - 1
            if self.Jz_std > 0:
                Jz_abs = torch.normal(self.Jz, self.Jz * self.Jz_std, size=(batch, L, L, n_layers))
                self.J[2] *= Jz_abs
            else:
                self.J[2] *= Jz
        else:
            self.J = J
        self.s = init_scale * (2 * torch.rand(batch, L, L, n_layers) - 1)
        self.x = init_scale * torch.rand(2, batch, L, L)

        self.s_x = np.arange(L)
        self.s_y = np.arange(L)
        self.s_x, self.s_y = np.meshgrid(self.s_x, self.s_y)
        self.J0_x = self.s_x
        self.J0_y = self.s_y - 0.5
        self.J1_x = self.s_x - 0.5
        self.J1_y = self.s_y

        self.site_idx = torch.arange(self.L * self.L).reshape(self.L, self.L)
        self.edges = torch.cat([torch.stack([self.site_idx[:, :-1].reshape(-1), self.site_idx[:, 1:].reshape(-1)], dim=1),
                                torch.stack([self.site_idx[:-1, :].reshape(-1), self.site_idx[1:, :].reshape(-1)], dim=1)], dim=0)
        self.coordinate = torch.stack(torch.meshgrid([torch.arange(self.L), torch.arange(self.L)]), dim=-1)\
            .reshape(self.L * self.L, 2).float()

        self.site_idx_3d = torch.arange(self.n_layers * self.L * self.L).reshape(self.n_layers, self.L, self.L)
        self.edges_3d = torch.cat([torch.stack([self.site_idx_3d[:, :, :-1].reshape(-1),
                                                self.site_idx_3d[:, :, 1:].reshape(-1)], dim=1),
                                   torch.stack([self.site_idx_3d[:, :-1, :].reshape(-1),
                                                self.site_idx_3d[:, 1:, :].reshape(-1)], dim=1),
                                   torch.stack([self.site_idx_3d[:-1, :, :].reshape(-1),
                                                self.site_idx_3d[1:, :, :].reshape(-1)], dim=1)], dim=0)

        self.s_last = None

        self.name_str = f'{L}_{gamma:.3f}_{Jz:.3f}'
        self.data_dir = data_dir
        self.snapshot_dir = 'snapshots'
        self.histogram_dir = 'histograms'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.histogram_dir, exist_ok=True)

    def f(self, s, x):
        # Open boundary condition
        sp = torch.nn.functional.pad(s, (0, 1, 0, 1, 0, 1), mode='constant', value=0)
        Jp = torch.nn.functional.pad(self.J, (0, 1, 0, 1, 0, 1), mode='constant', value=0)
        xp = torch.nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)

        # (2, batch, L+1, L+1)
        Jss = torch.stack([Jp[0, :, :, :, 0] * sp[:, :, :, 0] * sp[:, :, :, 0].roll(1, 1),
                           Jp[1, :, :, :, 0] * sp[:, :, :, 0] * sp[:, :, :, 0].roll(1, 2)], dim=0)
        C = (Jss[:, :, :-1, :-1] + 1) / 2
        dx = self.gamma * (C - self.delta)

        # (batch, L+1, L+1, L+1)
        ds = Jp[0] * sp.roll(1, 1) + Jp[0].roll(-1, 1) * sp.roll(-1, 1) \
           + Jp[1] * sp.roll(1, 2) + Jp[1].roll(-1, 2) * sp.roll(-1, 2) \
           + Jp[2] * sp.roll(1, 3) # + Jp[2].roll(-1, 3) * sp.roll(-1, 3)
        ds[:, :, :, 0] -= self.g * (xp[0] + xp[1] + xp[0].roll(-1, 1) + xp[1].roll(-1, 2)) * sp[:, :, :, 0]

        return ds[:, :-1, :-1, :-1], dx

    def Euler(self, s, x):
        ds, dx = self.f(s, x)
        s = (s + ds * self.dt).clamp(-1, 1)
        x = (x + dx * self.dt).clamp(0, 1)
        return s, x

    def RK4(self, s, x):
        k1, l1 = self.f(s, x)
        k2, l2 = self.f((s + 0.5 * k1 * self.dt).clamp(-1, 1), (x + 0.5 * l1 * self.dt).clamp(0, 1))
        k3, l3 = self.f((s + 0.5 * k2 * self.dt).clamp(-1, 1), (x + 0.5 * l2 * self.dt).clamp(0, 1))
        k4, l4 = self.f((s + k3 * self.dt).clamp(-1, 1), (x + l3 * self.dt).clamp(0, 1))
        s = (s + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6).clamp(-1, 1)
        x = (x + (l1 + 2 * l2 + 2 * l3 + l4) * self.dt / 6).clamp(0, 1)
        return s, x

    def LRO_metric(self, avalanche_stats):
        # (n_layers, 7)
        correlation_length, percolation_prob, slope, intercept, r, max_bin, max_dist = avalanche_stats.transpose(0, 1)
        # metric = -correlation_length * 5 / self.L + (slope + 2) ** 2 + (r.abs() - 1) ** 2 \
        #          + (max_bin - np.log10(self.L * self.L)) ** 2 + max_dist ** 2
        metric = -correlation_length / 10 + max_dist ** 2
        return metric  # (n_layers,)

    @torch.no_grad()
    def dynamics(self, n_steps, transient_steps, coarse_grain_steps=10, connection_dist=5, plot=False, init_ground_state=False):
        print(f'L = {self.L} gamma = {self.gamma:.3f} Jz = {self.Jz:.3f} Simulating transient steps: {transient_steps}')

        # compiled_step = torch.compile(self.RK4)
        compiled_step = self.RK4

        if init_ground_state:
            J_temp = self.J.clone()
            self.J[2] = 0  # Set Jz to 0, allow each layer to evolve towards ground state first
            for i in trange(500):
                self.s, self.x = compiled_step(self.s, self.x)
            self.J[2] = J_temp[2]  # Restore Jz

        for i in trange(transient_steps):
            self.s, self.x = compiled_step(self.s, self.x)

        print(f'L = {self.L} gamma = {self.gamma:.3f} Jz = {self.Jz:.3f} Simulating dynamics: {n_steps}')

        plot_steps = 10000

        spin_traj = []

        for i in trange(n_steps):
            if plot:
                if i % plot_steps == 0:
                    # flip = (self.s * s0) < 0
                    # self.plot_snapshot(flip[0], i // plot_steps)
                    self.plot_snapshot(self.s[0], i // plot_steps)
            self.s, self.x = compiled_step(self.s, self.x)
            spin_traj.append(self.s > 0)

        avalanche_stats_all = torch.zeros(self.n_layers, 7)
        metrics = torch.zeros(self.n_layers, 1)
        t_windows_all = torch.zeros(self.n_layers, 1)
        histograms = np.empty(self.n_layers, dtype=object)

        spin_traj = torch.stack(spin_traj, dim=0)
        for layer_idx in range(self.n_layers):
            spin_traj_i = spin_traj[:, :, :, :, layer_idx]
            window_steps = coarse_grain_steps
            t_idx = (torch.arange(int(n_steps / window_steps)) * window_steps).to(torch.int64)
            flip_traj_i = spin_traj_i[t_idx].clone().diff(dim=0)
            flip_traj_i = flip_traj_i.permute(1, 2, 3, 0).reshape(self.batch, self.L * self.L, -1)
            label, clusters, durations, Rs, Rs_t, is_percolating = \
                find_cluster_dense_connection(flip_traj_i, self.edges, self.coordinate, connection_dist)
            avalanche_stats_i, histogram_i = self.avalanche_stats([clusters, Rs, is_percolating])
            avalanche_stats_all[layer_idx] = torch.tensor(avalanche_stats_i, dtype=torch.float32)
            metric = self.LRO_metric(avalanche_stats_all[layer_idx].reshape(1, 7))
            metrics[layer_idx] = metric
            t_windows_all[layer_idx] = coarse_grain_steps * connection_dist
            histograms[layer_idx] = histogram_i

        avalanche_stats_all = torch.cat([avalanche_stats_all, metrics, t_windows_all], dim=1)

        if plot:
            self.plot_histogram_all_layers(histograms, f'{self.name_str}_{coarse_grain_steps}_{connection_dist}_all')
            for layer_idx in range(self.n_layers):
                histogram = histograms[layer_idx]
                if histogram is not None:
                    self.plot_histogram(histogram, f'{self.name_str}_'
                                                   f'tw_{coarse_grain_steps}_{connection_dist}_'
                                                   f'layer{layer_idx}')
        if self.save:
            torch.save(avalanche_stats_all, f'{self.data_dir}/avalanche_stats_{self.name_str}_{self.repeat_idx}.pt')
            # torch.save(mean_flip, f'{self.data_dir}/mean_flip_{self.name_str}_{self.repeat_idx}.pt')

        return

    def avalanche_stats(self, data):
        # cluster_sizes, _, Rs, _, is_percolating = data
        cluster_sizes, Rs, is_percolating = data
        mask = cluster_sizes > 0
        cluster_sizes = cluster_sizes[mask].float()
        Rs = Rs[mask]
        is_percolating = is_percolating[mask]

        if len(cluster_sizes) <= 2:
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], None

        log_cluster_sizes = cluster_sizes.log10()

        std = log_cluster_sizes.std()
        bin_width = 3.5 * std / (len(log_cluster_sizes) ** (1 / 3))
        bin_width = max(bin_width, 0.02)
        hist_min = 0
        hist_max = 10
        n_bins = int((hist_max - hist_min) / bin_width)
        n_bins = max(n_bins, 1)
        bins = bin_width * torch.arange(n_bins + 1)
        bins = (10 ** bins).int()
        bins = torch.unique(bins) - 0.5
        bin_centers = (bins[:-1] + bins[1:]) / 2

        idx = torch.searchsorted(bins, cluster_sizes) - 1
        idx[idx < 0] = 0
        idx[idx >= len(bin_centers)] = len(bin_centers) - 1
        hist = torch.zeros_like(bin_centers).index_add_(0, idx, torch.ones_like(cluster_sizes))

        hist = hist / (bins[1:] - bins[:-1])
        p_hist = hist / hist.sum()
        p_hist = p_hist[hist > 0]

        non_percolating_mask = ~is_percolating.bool()
        s = cluster_sizes[non_percolating_mask]
        Rs = Rs[non_percolating_mask]
        correlation_length = (((2 * Rs ** 2 * s ** 2).sum() / (s ** 2).sum()).sqrt()).cpu().numpy().item()
        percolation_prob = is_percolating.float().mean().cpu().numpy().item()

        bin_centers = bin_centers[hist > 0]
        hist = hist[hist > 0]

        histogram = [bin_centers, p_hist]

        # torch.save([bin_centers, p_hist], f'{self.histogram_dir}/{self.name_str}_{self.repeat_idx}.pt')

        try:
            # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
            slope, intercept, r, p, se = linregress(bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                                                    p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy())
            n_bins = len(bin_centers)
            max_bin = bin_centers[-1].log10().cpu().numpy().item()
            max_dist = bin_centers.log10().diff().max().cpu().numpy().item()
        except:
            slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]
            n_bins = 0
            max_bin = 0
            max_dist = 0

        stats = [correlation_length, percolation_prob, slope, intercept, r, max_bin, max_dist]

        return stats, histogram

    def plot_histogram(self, histogram, name):
        bin_centers, p_hist = histogram
        torch.save(histogram, f'{self.histogram_dir}/{name}.pt')
        try:
            # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
            slope, intercept, r, p, se = linregress(bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                                                    p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy())
        except:
            slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]
        bin_centers = bin_centers.cpu().numpy()
        p_hist = p_hist.cpu().numpy()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(bin_centers, p_hist, 'o', markersize=7, label='Data')
        ax.plot(bin_centers, 10 ** (slope * np.log10(bin_centers) + intercept), '--', label=f'~$s^{{{slope:.2f}}}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(0.7, 4000)
        # ax.set_ylim(2e-8, 1.5)
        ax.set_xlabel('Avalanche Size s')
        ax.set_ylabel('Probability P(s)')
        ax.legend()
        # ax.set_title(f'{self.name_str} {slope:.2f}')
        plt.savefig(f'{self.histogram_dir}/{name}.png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'{self.histogram_dir}/{name}.svg',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_histogram_all_layers(self, histograms, name):
        fig, ax = plt.subplots(figsize=(5, 4))
        for layer_idx, histogram_i in enumerate(histograms):
            try:
                bin_centers, p_hist = histogram_i
            except TypeError:
                continue
            if layer_idx == len(histograms) - 1:
                try:
                    # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
                    slope, intercept, r, p, se = linregress(
                        bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                        p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy())
                except:
                    slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]

            bin_centers = bin_centers.cpu().numpy()
            p_hist = p_hist.cpu().numpy()
            sc = ax.scatter(bin_centers, p_hist, s=15, alpha=0.5,
                            c=layer_idx * np.ones(len(bin_centers)), vmin=0, vmax=n_layers-1, cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.tick_params(labelsize=16)

        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(0.7, 4000)
        # ax.set_ylim(2e-8, 1.5)
        ax.set_xlabel('Avalanche Size s')
        ax.set_ylabel('Probability P(s)')
        # ax.legend()
        # ax.set_title(f'{self.name_str} {slope:.2f}')
        plt.savefig(f'{self.histogram_dir}/{name}.png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'{self.histogram_dir}/{name}.svg',
                    bbox_inches='tight', dpi=300)

        ax.plot(bin_centers, 10 ** (slope * np.log10(bin_centers) + intercept), '--', color='black')
        print(f'Fitted slope: {slope:.4f}, intercept: {intercept:.4f}')
        plt.savefig(f'{self.histogram_dir}/{name}_with_fit.png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'{self.histogram_dir}/{name}_with_fit.svg',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_snapshot(self, s, iteration):
        # s: (L, L, n_layers)
        n_layers = s.shape[-1]
        size = 1000 * (8 / self.L) ** 2

        for layer_idx in range(n_layers):
            si = s[:, :, layer_idx].cpu().numpy()
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            sc1 = ax1.scatter(self.s_x, self.s_y, s=size, c=si,
                        marker='s', cmap='bwr', vmin=-1, vmax=1)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(sc1, cax=cax)
            cbar.ax.tick_params(labelsize=16)

            ax1.set_aspect('equal')
            # ax1.set_title(f'Clause Function')
            ax1.axis('off')

            plt.savefig(f'{self.snapshot_dir}/s_{layer_idx}_{self.name_str}_{self.repeat_idx}_{iteration}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('--Jz_std', type=float, default=0.0, help='Standard deviation of Jz')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--n_intervals', type=int, default=1, help='Number of intervals to divide the dynamics data')
    parser.add_argument('--n_steps', type=int, default=4000, help='Number of steps for the dynamics simulation')
    parser.add_argument('--transient_steps', type=int, default=2000, help='Number of transient steps before dynamics')
    parser.add_argument('--coarse_grain_steps', type=int, default=2, help='Number of steps for coarse graining')
    parser.add_argument('--connection_dist', type=int, default=25, help='Connection distance for cluster finding')
    parser.add_argument('--plot', action='store_true', help='Whether to plot snapshots during simulation')
    parser.add_argument('--use_GPU', action='store_true', help='Whether to use GPU')
    parser.add_argument('--Ls', type=str, default='[16]')
    parser.add_argument('--gammas', type=str, default='[0.2]')
    parser.add_argument('--Jzs', type=str, default='[3.5]')
    parser.add_argument('--out', type=str, default='../test_0')
    parser.add_argument('--init_ground_state', action='store_true',
                        help='Whether to initialize the ground state before dynamics')
    args = parser.parse_args()
    index = args.index
    batch = args.batch
    Jz_std = args.Jz_std
    n_layers = args.n_layers
    n_intervals = args.n_intervals
    n_steps = args.n_steps
    transient_steps = args.transient_steps
    coarse_grain_steps = args.coarse_grain_steps
    connection_dist = args.connection_dist
    plot = args.plot
    use_GPU = args.use_GPU
    Ls = eval(args.Ls)
    gammas = eval(args.gammas)
    Jzs = eval(args.Jzs)
    out_dir = args.out
    init_ground_state = args.init_ground_state

    if use_GPU:
        torch.set_default_tensor_type(torch.cuda.FloatTensor
                                      if torch.cuda.is_available()
                                      else torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    g = 2
    delta = 0.75
    init_scale = 1

    # tw_reference = np.load('optimal_tw.npy')
    # Jz_reference = np.linspace(1.5, 5.5, 41)
    # log_gamma_reference = np.linspace(-2, 1, 46)
    # n_layers_reference = 11
    #
    # interpolators = []
    # for i in range(n_layers_reference):
    #     interpolators.append(RegularGridInterpolator((Jz_reference, log_gamma_reference), tw_reference[i, :, :]))

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    for L in Ls:
        for Jz in Jzs:
            for gamma in gammas:
                # t_windows = np.array([interpolators[i]((Jz, math.log10(gamma))) for i in range(n_layers_reference)])
                dt = 0.048 / gamma ** (1 / 3)
                model = Model(batch, L, n_layers, Jz, gamma, g, delta, dt, init_scale, J=None, repeat_idx=index,
                              data_dir=out_dir, save=True)
                model.dynamics(n_steps, transient_steps, plot=plot, init_ground_state=init_ground_state,
                               coarse_grain_steps=coarse_grain_steps, connection_dist=connection_dist)

