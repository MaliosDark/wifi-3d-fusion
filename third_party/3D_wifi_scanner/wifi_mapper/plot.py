import numpy as np

import matplotlib.pyplot as plt
import csv
import matplotlib.tri as tri
import math
from os import listdir
from os.path import isfile, join


def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


def create_plot(path, image_path, filename):
    x = []
    y = []
    z = []
    with open(path + filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            y.append(float(row[0]))
            x.append(float(row[1]))
            z.append(float(row[2]))

    xplotmax = max(x)
    yplotmax = max(y)

    y.append(float(30))
    x.append(float(30))
    z.append(float(-95))
    y.append(float(30))
    x.append(float(0))
    z.append(float(-50))

    latmin = min(x)
    latmax = max(x)
    lngmin = min(y)
    lngmax = max(y)


    npts = 150
    ngridx = 150
    ngridy = 150

    fig, ax = plt.subplots()
    xi = np.linspace(latmin, latmax, ngridx)
    yi = np.linspace(lngmin, lngmax, ngridy)

    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    ax.contour(xi, yi, zi, levels=20, linewidths=0.2, colors='k')
    cntr1 = ax.contourf(xi, yi, zi, levels=20, cmap="gist_rainbow")  # RdBu_r

    fig.colorbar(cntr1, ax=ax, orientation="horizontal").set_label("signal strength RSSI")
    ax.plot(x, y, 'ko', ms=0.5)
    ax.set(xlim=(latmin, xplotmax), ylim=(lngmin, yplotmax))
    ax.set_title("WiFi Map " + filename)
    ax.ticklabel_format(useOffset=False)

    ax.set_aspect(1)
    plt.savefig(path + image_path + filename + '.png', dpi=220)
    # plt.show()


path = 'stack_data/'
image_path = 'images/'
filename = 'csv_file_1637175707.csv'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print(onlyfiles)

for filename in onlyfiles:
    if filename != '.DS_Store' and filename != 'v':
        print(filename)
        create_plot(path, image_path, filename)
