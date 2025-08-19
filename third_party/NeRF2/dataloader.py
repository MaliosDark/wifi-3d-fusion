# -*- coding: utf-8 -*-
"""dataset processing and loading
"""
import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange



# def rssi2amplitude(rssi):
#     """convert rssi to amplitude
#     """
#     return 100 * 10 ** (rssi / 20)


# def amplitude2rssi(amplitude):
#     """convert amplitude to rssi
#     """
#     return 20 * np.log10(amplitude / 100)


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def split_dataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.npy')
        index = [i for i in range(np.load(csi_dir).shape[0])]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')




class Spectrum_dataset(Dataset):
    """spectrum dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        example_spt = imageio.imread(os.path.join(self.spectrum_dir, self.spt_names[0]))
        self.n_elevation, self.n_azimuth = example_spt.shape
        self.rays_per_spectrum = self.n_elevation * self.n_azimuth
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.nn_inputs, self.nn_labels = self.load_data()


    def __len__(self):
        return len(self.dataset_index) * self.rays_per_spectrum


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        train_inputs : tensor. [n_samples, 9]. The inputs for training
                  ray_o, ray_d, tx_pos
        """
        ## NOTE! Each spectrum will cost 1.2 MB of memory. Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 9)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## Load gateway position and orientation
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_info = yaml.safe_load(f)
            gateway_pos = gateway_info['gateway1']['position']
            gateway_orientation = gateway_info['gateway1']['orientation']

        ## Load transmitter position
        tx_pos = pd.read_csv(self.tx_pos_dir).values
        tx_pos = torch.tensor(tx_pos, dtype=torch.float32)

        ## Load data, each spectrum contains 90x360 pixels(rays)
        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            spectrum = imageio.imread(os.path.join(self.spectrum_dir, idx + '.png')) / 255
            spectrum = torch.tensor(spectrum, dtype=torch.float32).view(-1, 1)
            ray_o, ray_d = self.gen_rays_spectrum(gateway_pos, gateway_orientation)
            tx_pos_i = torch.tile(tx_pos[int(idx)-1], (self.rays_per_spectrum,)).reshape(-1,3)  # [n_rays, 3]
            nn_inputs[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :9] = \
                torch.cat([ray_o, ray_d, tx_pos_i], dim=1)
            nn_labels[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :] = spectrum

        return nn_inputs, nn_labels


    def gen_rays_spectrum(self, gateway_pos, gateway_orientation):
        """generate sample rays origin at gateway with resolution given by spectrum

        Parameters
        ----------
        azimuth : int. The number of azimuth angles
        elevation : int. The number of elevation angles

        Returns
        -------
        r_o : tensor. [n_rays, 3]. The origin of rays
        r_d : tensor. [n_rays, 3]. The direction of rays, unit vector
        """

        azimuth = torch.linspace(1, 360, self.n_azimuth) / 180 * np.pi
        elevation = torch.linspace(1, 90, self.n_elevation) / 180 * np.pi
        azimuth = torch.tile(azimuth, (self.n_elevation,))  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
        elevation = torch.repeat_interleave(elevation, self.n_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

        x = 1 * torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
        y = 1 * torch.cos(elevation) * torch.sin(azimuth)
        z = 1 * torch.sin(elevation)

        r_d = torch.stack([x, y, z], dim=0)  # [3, n_rays] 3D direction of rays in gateway coordinate
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        r_d = R @ r_d  # [3, n_rays] 3D direction of rays in world coordinate
        gateway_pos = torch.tensor(gateway_pos, dtype=torch.float32)
        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3)  # [n_rays, 3]

        return r_o, r_d.T





class BLE_dataset(Dataset):
    """ble dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_position.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load gateway position
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_pos_dict = yaml.safe_load(f)
            self.gateway_pos = torch.tensor([pos for pos in gateway_pos_dict.values()], dtype=torch.float32)
            self.gateway_pos = self.gateway_pos / scale_worldsize
            self.n_gateways = len(self.gateway_pos)

        # Load transmitter position
        self.tx_poses = torch.tensor(pd.read_csv(tx_pos_dir).values, dtype=torch.float32)
        self.tx_poses = self.tx_poses / scale_worldsize

        # Load gateway received RSSI
        self.rssis = torch.tensor(pd.read_csv(self.rssi_dir).values, dtype=torch.float32)

        self.nn_inputs, self.nn_labels = self.load_data()


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        nn_inputs : tensor. [n_samples, 978]. The inputs for training
                    tx_pos:3, ray_o:3, ray_d:9x36x3,
        nn_labels : tensor. [n_samples, 1]. The RSSI labels for training
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## generate rays origin at gateways
        gateways_ray_o, gateways_rays_d = self.gen_rays_gateways()

        ## Load data
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):
            rssis = self.rssis[idx]
            tx_pos = self.tx_poses[idx].view(-1)  # [3]
            for i_gateway, rssi in enumerate(rssis):
                if rssi != -100:
                    gateway_ray_o = gateways_ray_o[i_gateway].view(-1)  # [3]
                    gateway_rays_d = gateways_rays_d[i_gateway].view(-1)  # [n_rays x 3]
                    nn_inputs[data_counter] = torch.cat([tx_pos, gateway_ray_o, gateway_rays_d], dim=-1)
                    nn_labels[data_counter] = rssi
                    data_counter += 1

        nn_labels = rssi2amplitude(nn_labels)

        return nn_inputs, nn_labels


    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """


        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_gateways, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o = self.gateway_pos.unsqueeze(1) # [21, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d


    def __len__(self):
        rssis = self.rssis[self.dataset_index]
        return torch.sum(rssis != -100)

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]




class CSI_dataset(Dataset):

    def __init__(self, datadir, indexdir, scale_worldsize=1):
        """ datasets [datalen*8, up+down+r_o+r_d] --> [datalen*8, 26+26+3+36*3]
        """
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, 'csidata.npy')
        self.bs_pos_dir = os.path.join(datadir, 'base-station.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load base station position
        with open(os.path.join(self.bs_pos_dir)) as f:
            bs_pos_dict = yaml.safe_load(f)
            self.bs_pos = torch.tensor([bs_pos_dict["base_station"]], dtype=torch.float32).squeeze()
            self.bs_pos = self.bs_pos / scale_worldsize
            self.n_bs = len(self.bs_pos)

        # load CSI data
        csi_data = torch.from_numpy(np.load(self.csidata_dir))  #[N, 8, 52]
        csi_data = self.normalize_csi(csi_data)
        uplink, downlink = csi_data[..., :26], csi_data[..., 26:]
        up_real, up_imag = torch.real(uplink), torch.imag(uplink)
        down_real, down_imag = torch.real(downlink), torch.imag(downlink)
        self.uplink = torch.cat([up_real, up_imag], dim=-1)    # [N, 8, 52]
        self.downlink = torch.cat([down_real, down_imag], dim=-1)    # [N, 8, 52]
        self.uplink = rearrange(self.uplink, 'n g c -> (n g) c')    # [N*8, 52]
        self.downlink = rearrange(self.downlink, 'n g c -> (n g) c')    # [N*8, 52]

        self.nn_inputs, self.nn_labels = self.load_data()


    def normalize_csi(self, csi):
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi):
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 1027]. The inputs for training
                    uplink: 52 (26 real; 26 imag), ray_o: 3, ray_d: 9x36x3, n_samples = n_dataset * n_bs
        nn_labels : tensor. [n_samples, 52]. The downlink channels as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 52+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 52)), dtype=torch.float32)

        ## generate rays origin at gateways
        bs_ray_o, bs_rays_d = self.gen_rays_gateways()
        bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        ## Load data
        for data_counter, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            bs_uplink = self.uplink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            bs_downlink = self.downlink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            nn_inputs[data_counter*self.n_bs: (data_counter+1)*self.n_bs] = torch.cat([bs_uplink, bs_ray_o, bs_rays_d], dim=-1) # [n_bs, 52+3+3*36*9]
            nn_labels[data_counter*self.n_bs: (data_counter+1)*self.n_bs]  = bs_downlink
        return nn_inputs, nn_labels

    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_bs, 1, 3]. The origin of rays
        r_d : tensor. [n_bs, n_rays, 3]. The direction of rays, unit vector
        """
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_bs, self.beta_res * self.alpha_res, 3])  # [n_bs, 9*36, 3]
        r_o = self.bs_pos.unsqueeze(1) # [n_bs, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def __len__(self):
        return len(self.dataset_index) * self.n_bs


dataset_dict = {"rfid": Spectrum_dataset, "ble": BLE_dataset, "mimo": CSI_dataset}
