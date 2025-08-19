# -*- coding: utf-8 -*-
"""painter for data
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def paint_spectrum(spectrum, save_path=None):

    spectrum = spectrum.numpy().reshape(90, 360)
    plt.imsave(save_path, spectrum, cmap='jet')
    spectrum = np.flipud(spectrum)
    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cax = ax.pcolormesh(theta, r, spectrum.T, cmap='jet', shading='flat')
    ax.axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)


def paint_spectrum_compare(pred_spectrum, gt_spectrum, save_path=None):

    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    cax1 = axs[0].pcolormesh(theta, r, np.flipud(pred_spectrum).T, cmap='viridis', shading='flat')
    axs[0].axis('off')

    cax2 = axs[1].pcolormesh(theta, r, np.flipud(gt_spectrum).T, cmap='viridis', shading='flat')
    axs[1].axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def paint_location(loc_path, save_path):


    all_loc = np.loadtxt(os.path.join(loc_path, 'tx_pos.csv'), delimiter=',', skiprows=1)
    train_index = np.loadtxt(os.path.join(loc_path, 'train_index.txt'), dtype=int)
    test_index = np.loadtxt(os.path.join(loc_path, 'test_index.txt'), dtype=int)
    train_loc = all_loc[train_index-1]
    test_loc = all_loc[test_index-1]
    plt.scatter(train_loc[:, 0], train_loc[:, 1], c='b', label='train',s=0.1)
    plt.scatter(test_loc[:, 0], test_loc[:, 1], c='r', label='test',s=0.1)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loc.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    loc_path = "data/s23/"
    save_path = "data/s23/"
    paint_location(loc_path, save_path)