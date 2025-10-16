import os
import argparse
import math
import numpy as np
import torch
from tqdm import tqdm, trange
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plt_config
import json
import pandas as pd

from cluster_finding_dense import find_cluster_dense_connection
from avalanche_extraction import avalanche_extraction

matplotlib.use('Agg')
color = plt.rcParams['axes.prop_cycle'].by_key()['color']



# torch.set_default_tensor_type(torch.FloatTensor)


class Model:
    def __init__(self, batch, L, n_layers=11, Jz=1, gamma=0.2, g=2, delta=0.75, dt=0.05, init_scale=1,
                 J=None, Jz_std=0, repeat_idx=0, save=True, data_dir='data_phase_diagram_25'):
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
        self.snapshot_dir = 'figures_phase_diagram_25'
        self.histogram_dir = 'histograms_phase_diagram_25'
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

    @torch.no_grad()
    def dynamics(self, n_steps, transient_steps, coarse_grain_steps=1, connection_dist=10, plot=False, init_ground_state=False):
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
            #if plot:
            #    if i % plot_steps == 0:
            #        # flip = (self.s * s0) < 0
            #        # self.plot_snapshot(flip[0], i // plot_steps)
            #        self.plot_snapshot(self.s[0], i // plot_steps)
            self.s, self.x = compiled_step(self.s, self.x)
            spin_traj.append(self.s > 0)

        avalanche_stats_all = torch.zeros(self.n_layers, 5)
        metrics = torch.zeros(self.n_layers, 1)
        t_windows_all = torch.zeros(self.n_layers, 1)
        histograms = np.empty(self.n_layers, dtype=object)

        spin_traj = torch.stack(spin_traj, dim=0)
        use_optimal_window = False
        for layer_idx in range(self.n_layers):
            spin_traj_i = spin_traj[:, :, :, :, layer_idx]
            window_steps = coarse_grain_steps
            t_idx = (torch.arange(int(n_steps / window_steps)) * window_steps).to(torch.int64)
            flip_traj_i = spin_traj_i[t_idx].clone().diff(dim=0)

            #Additional code to extract optimal time window from existing database, if no explicit window value is provided
            if connection_dist is None:
                connection_dist = find_optimal_window(layer_idx, self.gamma, self.Jz)
                use_optimal_window = True

            '''#Avalanche extraction (from trajectories, with YH's original cluster-finding code)
            flip_traj_i = flip_traj_i.permute(1, 2, 3, 0).reshape(self.batch, self.L * self.L, -1)
            _, clusters, _, Rs, _, is_percolating = \
                find_cluster_dense_connection(flip_traj_i, self.edges, self.coordinate, connection_dist)'''

            #Avalanche extraction (from trajectories, with Chesson's code)
            flip_traj_i = flip_traj_i.permute(1, 0, 2, 3)
            temp_clusters = []
            for j in range(batch):
                temp_clusters_j = avalanche_extraction(flip_traj_i[j], connection_dist, self.dt)
                with open(f'{self.data_dir}/{self.name_str}_{coarse_grain_steps}_{connection_dist}_{layer_idx}_clusters_sizes_{j}.json', 'w') as f:
                    json.dump(temp_clusters_j, f)
                temp_clusters.append(temp_clusters_j)
            clusters = [item for sublist in temp_clusters for item in sublist]

            '''#Avalanche extraction (from .json files)
            temp_clusters = []
            for j in range(batch):
                with open(f'{self.data_dir}/{self.name_str}_{coarse_grain_steps}_{connection_dist}_{layer_idx}_clusters_sizes_{j}.json', 'r') as f:
                    temp_clusters_j = json.load(f)
                temp_clusters.append(temp_clusters_j)
            clusters = [item for sublist in temp_clusters for item in sublist]'''

            avalanche_stats_i, histogram_i, phase_i = self.avalanche_stats(torch.tensor(clusters))
            avalanche_stats_all[layer_idx] = torch.tensor(avalanche_stats_i, dtype=torch.float32)
            t_windows_all[layer_idx] = coarse_grain_steps * connection_dist
            histograms[layer_idx] = histogram_i
            with open(f'{self.data_dir}/{self.name_str}_{coarse_grain_steps}_{connection_dist}_{layer_idx}_phase.json', 'w') as f:
                json.dump(phase_i, f)
            
            if use_optimal_window:
                connection_dist = None

        avalanche_stats_all = torch.cat([avalanche_stats_all, t_windows_all], dim=1)

        if plot:
            if any([len(element[0]) > 0 for element in histograms]):
                self.plot_histogram_all_layers(histograms, f'{self.name_str}_{coarse_grain_steps}_{connection_dist}_all')
            for layer_idx in range(self.n_layers):
                histogram = histograms[layer_idx]
                if len(histogram[0]) > 0:
                    self.plot_histogram(histogram, f'{self.name_str}_'
                                                   f'tw_{coarse_grain_steps}_{connection_dist}_'
                                                   f'layer{layer_idx}')
        if self.save:
            torch.save(avalanche_stats_all, f'{self.data_dir}/avalanche_stats_{self.name_str}_{self.repeat_idx}.pt')
            # torch.save(mean_flip, f'{self.data_dir}/mean_flip_{self.name_str}_{self.repeat_idx}.pt')

        return histograms

    def avalanche_stats(self, data):
        cluster_sizes = data
        mask = cluster_sizes > 0
        cluster_sizes = cluster_sizes[mask].float()

        if len(cluster_sizes) <= 2:
            print("Phase: no_dynamics")
            return [np.nan, np.nan, np.nan, np.nan, np.nan], [torch.tensor([], dtype=torch.float64), torch.tensor([], dtype=torch.float64)], "no_dynamics"

        '''#Yuanhang's distribution binning approach
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

        bin_centers = bin_centers[hist > 0]
        hist = hist[hist > 0]

        histogram = [bin_centers, p_hist]'''


        #Chesson's distribution binning approach (identical to previous paper)
        cluster_sizes = [int(element) for element in cluster_sizes.tolist()]
        counts = np.zeros(max(cluster_sizes))
        for i in range(len(cluster_sizes)):
            counts[cluster_sizes[i] - 1] += 1 #offsets, since indexing starts at 0, but the first element in "counts" should correspond to the number of avalanches of size 1

        #Choose a logarithmic binning approach which gives reasonable results (see manuscript for a discussion of this particular binning approach)
        custom_bins = [i for i in range(0, 100, 2)] + [i for i in range(100, 1000, 20)] + [i for i in range(1000, 10000, 200)] + [i for i in range(10000, 100000, 2000)]
        custom_bins = [my_bin for my_bin in custom_bins if my_bin <= max(cluster_sizes)] #truncates empty bins
        
        p_hist, bin_edges = np.histogram(cluster_sizes, bins=custom_bins, density=True)
        bin_centers = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges) - 1)])

        bin_centers = torch.tensor(bin_centers)
        p_hist = torch.tensor(p_hist)

        if len(bin_centers) == 0:
            print("Phase: no_dynamics")
            return [np.nan, np.nan, np.nan, np.nan, np.nan], [torch.tensor([], dtype=torch.float64), torch.tensor([], dtype=torch.float64)], "no_dynamics"
        
        histogram = [bin_centers, p_hist]


        #Fits distribution to a power-law decay
        try:
            '''slope, intercept, r, p, se = linregress(bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                                                    p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy()) #YH's fit'''
            slope, intercept, r, p, se = linregress(bin_centers[p_hist > 0.0][1:75].log10().cpu().numpy(),
                                                    p_hist[p_hist > 0.0][1:75].log10().cpu().numpy()) #Chesson's fit
            n_bins = len(bin_centers)
            max_bin = bin_centers[-1].log10().cpu().numpy().item()
            max_dist = bin_centers.log10().diff().max().cpu().numpy().item()
        except:
            slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]
            n_bins = 0
            max_bin = 0
            max_dist = 0
        
        #Characterizes phase
        phase = "outside_provided_phases"
        has_LRO = (slope >= -2.5) & (max_bin > np.log10(L**2) - 1.5) & (max_dist <= 1)
        rigid = (max_bin > np.log10(L**2) - 1.5) & (max_dist > 1)
        #LRO_to_rigid = (max_bin > np.log10(L**2) - 0.5) & (max_dist < 1) & (max_dist > 0.5)
        #LRO_to_SRO = (slope < -1.25) & (slope > -2.75) & (max_bin < np.log10(L**2) - 0.5) \
        #            & (max_bin > np.log10(L**2) - 1.5)
        SRO = (max_bin <= np.log10(L**2) - 1.5) & (max_bin > 0.5)
        no_dynamics = (max_bin <= 0.5)
        if has_LRO:
            phase = "has_LRO"
        elif rigid:
            phase = "rigid"
        #elif LRO_to_rigid:
        #    phase = "LRO_to_rigid"
        #elif LRO_to_SRO:
        #    phase = "LRO_to_SRO"
        elif SRO:
            phase = "SRO"
        elif no_dynamics:
            phase = "no_dynamics"
        print("Phase: " + str(phase))

        stats = [slope, intercept, r, max_bin, max_dist]

        return stats, histogram, phase

    def plot_histogram(self, histogram, name):
        bin_centers, p_hist = histogram
        #torch.save(histogram, f'{self.histogram_dir}/{name}.pt')
        try:
            # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
            '''slope, intercept, r, p, se = linregress(bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                                                    p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy()) #YH's fit'''
            slope, intercept, r, p, se = linregress(bin_centers[p_hist > 0.0][1:75].log10().cpu().numpy(),
                                                    p_hist[p_hist > 0.0][1:75].log10().cpu().numpy()) #Chesson's fit
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
        #plt.savefig(f'{self.histogram_dir}/{name}.svg',
        #            bbox_inches='tight', dpi=300)
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
                    '''slope, intercept, r, p, se = linregress(
                        bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                        p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy()) #YH's fit'''
                    slope, intercept, r, p, se = linregress(bin_centers[p_hist > 0.0][1:75].log10().cpu().numpy(),
                                                    p_hist[p_hist > 0.0][1:75].log10().cpu().numpy()) #Chesson's fit
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
        #plt.savefig(f'{self.histogram_dir}/{name}.svg',
        #            bbox_inches='tight', dpi=300)

        if not np.isnan(slope):
            ax.plot(bin_centers, 10 ** (slope * np.log10(bin_centers) + intercept), '--', color='black', label=rf'$\sim s^{{{slope:.2f}}}$')
            print(f'Fitted slope: {slope:.4f}, intercept: {intercept:.4f}')
            ax.legend()
            plt.savefig(f'{self.histogram_dir}/{name}_with_fit.png',
                        bbox_inches='tight', dpi=300)
            #plt.savefig(f'{self.histogram_dir}/{name}_with_fit.svg',
            #            bbox_inches='tight', dpi=300)
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

def plot_histogram_all_sizes(all_histograms, name, Ls):
    for layer_idx in range(len(all_histograms[0])):
        fig, ax = plt.subplots(figsize=(5, 4))
        for L_i, histogram_L in enumerate(all_histograms):
            try:
                bin_centers, p_hist = histogram_L[layer_idx]
            except TypeError:
                continue
            if L_i == len(all_histograms) - 1:
                try:
                    # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
                    '''slope, intercept, r, p, se = linregress(
                        bin_centers[:int(0.6 * len(bin_centers))].log10().cpu().numpy(),
                        p_hist[:int(0.6 * len(bin_centers))].log10().cpu().numpy()) #YH's fit'''
                    slope, intercept, r, p, se = linregress(bin_centers[p_hist > 0.0][1:75].log10().cpu().numpy(),
                                                    p_hist[p_hist > 0.0][1:75].log10().cpu().numpy()) #Chesson's fit
                except:
                    slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]

            bin_centers = bin_centers.cpu().numpy()
            p_hist = p_hist.cpu().numpy()
            if len(histogram_L[layer_idx][0]) == 0:
                pass
            else:
                sc = ax.scatter(bin_centers, p_hist, s=15, alpha=0.5, label=rf'$N = {{{Ls[L_i]}}}^2$')

        if not np.isnan(slope):
            ax.plot(bin_centers, 10 ** (slope * np.log10(bin_centers) + intercept), '--', color='black', alpha=0.5, label=rf'$\sim s^{{{slope:.2f}}}$')
            ax.legend(fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(0.7, 4000)
        # ax.set_ylim(2e-8, 1.5)
        ax.set_xlabel('Avalanche Size s')
        ax.set_ylabel('Probability P(s)')
        # ax.set_title(f'{self.name_str} {slope:.2f}')
        plt.savefig(f'histograms_phase_diagram_25/{name}_layer{layer_idx}.png',
                    bbox_inches='tight', dpi=300)
        #plt.savefig(f'histograms_phase_diagram_25/{name}_layer{layer_idx}.svg',
        #            bbox_inches='tight', dpi=300)
        plt.close()

        '''if not np.isnan(slope):
            fig_finite, ax_finite = plt.subplots(figsize=(5, 4))
            for L_i, histogram_L in enumerate(all_histograms):
                bin_centers, p_hist = histogram_L[layer_idx]
                bin_centers = bin_centers.cpu().numpy()
                p_hist = p_hist.cpu().numpy()
                if len(histogram_L[layer_idx][0]) == 0:
                    pass
                else:
                    sc_finite = ax_finite.scatter(bin_centers/(Ls[L_i]**2.0), p_hist*(bin_centers**(-1*slope)), s=15, alpha=0.5)
            ax_finite.set_xscale('log')
            ax_finite.set_yscale('log')
            ax_finite.set_xlabel(r'$s/L^2$')
            ax_finite.set_ylabel(rf'$s^{{{-1*slope:.2f}}} P(s)$')
            plt.savefig(f'histograms_phase_diagram_25/{name}_layer{layer_idx}_finite.png',
                        bbox_inches='tight', dpi=300)
            #plt.savefig(f'histograms_phase_diagram_25/{name}_layer{layer_idx}_finite.svg',
            #            bbox_inches='tight', dpi=300)
            plt.close()'''

def find_optimal_window(layer_idx, gamma, Jz):
    f = f'optimal_time_windows/t_windows_layer{layer_idx}.xlsx'
    df = pd.read_excel(f, sheet_name='Sheet1')
    gamma_list = [df.columns[j+1] for j in range(len(df.columns)-1)]
    gamma_list.insert(0, 0.0)
    Jz_list = [element[0] for element in df.values]

    proper_gamma = min(gamma_list, key=lambda x:abs(x-gamma))
    proper_Jz = min(Jz_list, key=lambda x:abs(x-Jz))
    proper_gamma_index = gamma_list.index(proper_gamma)
    proper_Jz_index = Jz_list.index(proper_Jz)
    connection_dist = df.values[proper_Jz_index][proper_gamma_index]
    return connection_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=25)
    parser.add_argument('--Jz_std', type=float, default=0.0, help='Standard deviation of Jz')
    parser.add_argument('--n_layers', type=int, default=11, help='Number of layers in the model')
    parser.add_argument('--n_steps', type=int, default=200, help='Number of steps for the dynamics simulation')
    parser.add_argument('--transient_steps', type=int, default=2000, help='Number of transient steps before dynamics')
    parser.add_argument('--coarse_grain_steps', type=int, default=1, help='Number of steps for coarse graining')
    parser.add_argument('--connection_dists', type=str, default='[None]', help='Connection distance for cluster finding')
    parser.add_argument('--plot', type=bool, default=True, help='Whether to plot snapshots during simulation')
    parser.add_argument('--use_GPU', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--Ls', type=str, default='[64]')
    parser.add_argument('--gammas', type=str, default=f'{[element for element in np.logspace(-2, 1, num=24)]}')
    parser.add_argument('--Jzs', type=str, default=f'{[round(element, 1) for element in np.linspace(1.5, 5.5, num=24)]}')
    parser.add_argument('--out', type=str, default='data_phase_diagram_25')
    parser.add_argument('--init_ground_state', action='store_true', help='Whether to initialize the ground state before dynamics')
    args = parser.parse_args()
    index = args.index
    batch = args.batch
    Jz_std = args.Jz_std
    n_layers = args.n_layers
    n_steps = args.n_steps
    transient_steps = args.transient_steps
    coarse_grain_steps = args.coarse_grain_steps
    connection_dists = eval(args.connection_dists)
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

    os.makedirs(out_dir, exist_ok=True)
    for Jz in Jzs:
        for gamma in gammas:
            for connection_dist in connection_dists:
                all_histograms = []
                for L in Ls:
                    dt = 0.048 / gamma ** (1 / 3)
                    model = Model(batch, L, n_layers, Jz, gamma, g, delta, dt, init_scale, J=None, repeat_idx=index,
                                data_dir=out_dir, save=True)
                    histograms_L = model.dynamics(n_steps, transient_steps, plot=plot, init_ground_state=init_ground_state,
                                coarse_grain_steps=coarse_grain_steps, connection_dist=connection_dist)
                    all_histograms.append(histograms_L)
                #if plot:
                #    if any([len(element[0]) > 0 for element in all_histograms]):
                #        plot_histogram_all_sizes(all_histograms, f'{Ls}_{gamma:.3f}_{Jz:.3f}_tw_{coarse_grain_steps}_{connection_dist}', Ls)

#Loop for generating phase diagram
    for L in Ls:
        for layer_idx in range(2):#n_layers):
            phase_diagram_data = [["" for _ in range(len(gammas))] for _ in range(len(Jzs))]
            phase_diagram_colors = [["" for _ in range(len(gammas))] for _ in range(len(Jzs))]
            for J_index, Jz in enumerate(Jzs):
                for g_index, gamma in enumerate(gammas):
                    connection_dist = find_optimal_window(layer_idx, gamma, Jz)
                    with open(f'data_phase_diagram_25/{L}_{gamma:.3f}_{Jz:.3f}_{coarse_grain_steps}_{connection_dist}_{layer_idx}_phase.json', 'r') as f:
                        phase = json.load(f)
                        phase_diagram_data[J_index][g_index] = phase
                        if phase == "rigid":
                            color = 1
                        elif phase == "LRO_to_rigid":
                            color = 2
                        elif phase == "has_LRO":
                            color = 3
                        elif phase == "LRO_to_SRO":
                            color = 4
                        elif phase == "SRO":
                            color = 5
                        elif phase == "no_dynamics":
                            color = 6
                        elif phase == "outside_provided_phases":
                            color = 7
                        phase_diagram_colors[J_index][g_index] = color

            phase_diagram_colors = np.array(phase_diagram_colors)
            
            colors = ['royalblue', 'dodgerblue', 'deepskyblue', 'coral', 'orangered', 'black', 'white']
            cmap = ListedColormap(colors)
            bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
            norm = BoundaryNorm(bounds, cmap.N)

            #Interpolation
            log_x_grid, y_grid = np.mgrid[np.log10(gammas[0]):np.log10(gammas[-1]):100j,
                                          Jzs[0]:Jzs[-1]:100j]
            log_x_grid = 10**log_x_grid
            phase_diagram_colors = phase_diagram_colors.flatten()
            xy_array = np.array([[x, y] for y in Jzs for x in gammas])
            interpolated_phase_diagram_colors = griddata(xy_array, phase_diagram_colors, (log_x_grid, y_grid), method='linear')

            fig, ax = plt.subplots(figsize=(8, 8))

            mesh = ax.pcolormesh(log_x_grid, y_grid, interpolated_phase_diagram_colors, cmap=cmap, norm=norm)

            ax.set_xscale('log')
            ax.set_xlim(left=gammas[0], right=gammas[-1])
            ax.set_ylim(bottom=Jzs[0], top=Jzs[-1])
            ax.set_xlabel(r'$\gamma$', fontsize=32)
            ax.set_ylabel(r'$J^{\perp}$', fontsize=32)
            ax.tick_params(labelsize=22) 
            ax.set_title(f'Layer {layer_idx}', fontsize=28)

            plt.savefig(f'figures_phase_diagram_25/{L}_{coarse_grain_steps}_{layer_idx}_phase_diagram.png', dpi=300, bbox_inches='tight')
            plt.close()
