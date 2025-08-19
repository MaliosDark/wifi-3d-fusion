# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sconst
import torch
import imageio.v2 as imageio

# x,y,z coordinates of 16 antennas, customized for your own antenna array
ANT_LOC = [[-0.24, -0.24, 0], [-0.08, -0.24, 0], [0.08, -0.24, 0], [0.24, -0.24, 0],
           [-0.24, -0.08, 0], [-0.08, -0.08, 0], [0.08, -0.08, 0], [0.24, -0.08, 0],
           [-0.24,  0.08, 0], [-0.08,  0.08, 0], [0.08,  0.08, 0], [0.24,  0.08, 0],
           [-0.24,  0.24, 0], [-0.08,  0.24, 0], [0.08,  0.24, 0], [0.24,  0.24, 0]]

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

class Bartlett():
    """ Class to generate Spatial Spectrum using Bartlett Algorithm. """
    def __init__(self, frequency=920e6):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.antenna_loc = torch.tensor(ANT_LOC, dtype=torch.float32).T  # 3x16
        self.lamda = sconst.c / frequency
        self.theory_phase = self._calculate_theory_phase().to(self.device)

    def _calculate_theory_phase(self):
        """ Calculates theoretical phase difference over both azimuthal and elevation angle. """
        azimuth = torch.linspace(0, 359, 360) / 180 * np.pi
        elevation = torch.linspace(1, 90, 90) / 180 * np.pi

        # azimuth[0,1,..0,1..], elevation [0,0,..1,1..]
        elevation_grid, azimuth_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
        azimuth_grid = azimuth_grid.flatten()
        elevation_grid = elevation_grid.flatten()

        theory_dis_diff = (self.antenna_loc[0,:].unsqueeze(-1) * torch.cos(azimuth_grid) * torch.cos(elevation_grid) +
                        self.antenna_loc[1,:].unsqueeze(-1) * torch.sin(azimuth_grid) * torch.cos(elevation_grid))
        theory_phase = -2 * np.pi * theory_dis_diff / self.lamda
        return theory_phase.T

    def gen_spectrum(self, phase_measurements):
        """ Generates spatial spectrum from phase measurements. """
        phase_measurements = torch.tensor(phase_measurements, dtype=torch.float32).to(self.device)
        delta_phase = self.theory_phase - phase_measurements.reshape(1, -1)   # (360x90,16) - 1x16
        phase_sum = torch.exp(1j * delta_phase).sum(1) / self.antenna_loc.shape[1]
        spectrum = normalize(torch.abs(phase_sum)).view(90, 360).cpu().numpy()
        return spectrum


if __name__ == '__main__':

    sample_phase = [-1.886,-1.923,-2.832,-1.743,
                -1.751,-1.899,-2.370,-3.113,
                -2.394,-2.464,2.964,-2.904,
                -1.573,-2.525,-3.039,-2.839]
    worker = Bartlett()
    spectrum = worker.gen_spectrum(sample_phase)
    spectrum = (spectrum * 255).astype(np.uint8)

    imageio.imsave('spectrum.png', spectrum)