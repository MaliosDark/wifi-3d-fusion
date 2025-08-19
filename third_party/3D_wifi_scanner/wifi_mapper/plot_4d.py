import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.tri as tri
import math
from os import listdir
from os.path import isfile, join


x = []
y = []
z = []
c = []
s = []
o = []

path = 'stack_data/'
filename = 'csv_file_1637175707.csv'
fileindex = 0

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles = sorted(onlyfiles)

for filename in onlyfiles:
    if filename != '.DS_Store':
        with open(path + filename, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if float(row[2]) < -54.0:
                    y.append(float(row[0]))
                    x.append(float(row[1]))
                    z.append(float(fileindex))
                    c.append(float(row[2]))
                    s.append((-float(row[2]) - 57))
                    # o.append(-float(row[2]) / 200)
        fileindex += 1

print(min(s))
fig = go.Figure(data=[go.Scatter3d(x=x,
                                   y=y,
                                   z=z,
                                   marker=dict(
                                       size=s,
                                       color=c,  # set color to an array/list of desired values
                                       colorscale='phase',  # choose a colorscale
                                       opacity=0.4
                                   ),
                                   mode='markers')]

                )
fig.show()
