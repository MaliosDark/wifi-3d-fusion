import json
import requests
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.tri as tri

filetime = str(round(time.time()))
logfile = 'stack_data/csv_file_' + filetime + '.csv'
f = open(logfile, 'w')
writer = csv.writer(f)

UDP_SEND_IP_MOTOR_Y = "192.168.1.177" # change to ethersweep IP of Y axis motor 
UDP_SEND_IP_MOTOR_X = "192.168.1.178" # change to ethersweep IP of X axis motor 
UDP_SEND_PORT = 8888

url = 'http://192.168.1.103/rssi' # change to NODEMCU IP

resp = requests.get(url=url)
data = resp.json()

x_measurements = []


def drive_motor(steps, speed, direction, stepmode, motor_ip):
    json_data = json.dumps({'steps': steps, 'speed': speed, 'direction': direction, 'stepmode': stepmode})
    message = json_data.encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message, (motor_ip, UDP_SEND_PORT))
    print(message)


def getSignal():
    resp = requests.get(url=url)
    data = resp.json()
    print(data)
    return data['signal']


x_images = 24
y_images = 20
x_steps_per_step = 400
y_steps_per_step = 1000


# check if grid is large enough to make full measurement
#drive_motor(x_steps_per_step * x_images, 80, 0, 8, UDP_SEND_IP_MOTOR_X)
#drive_motor(y_steps_per_step * y_images, 200, 1, 4, UDP_SEND_IP_MOTOR_Y)
input("Press Enter to continue...")
#drive_motor(x_steps_per_step * x_images, 80, 1, 8, UDP_SEND_IP_MOTOR_X)
#drive_motor(y_steps_per_step * y_images, 200, 0, 4, UDP_SEND_IP_MOTOR_Y)
input("Press Enter to start scan...")


for y in range(y_images):
    drive_motor(y_steps_per_step, 200, 1, 4, UDP_SEND_IP_MOTOR_Y)
    time.sleep(1)

    for x in range(x_images):
        drive_motor(x_steps_per_step, 100, 0, 8, UDP_SEND_IP_MOTOR_X)
        time.sleep(0.1)
        signal = getSignal()
        x_measurements.append(signal)
        data = []
        data.append(y)
        data.append(x)
        data.append(signal)
        writer.writerow(data)

    drive_motor(x_steps_per_step * x_images, 100, 1, 8, UDP_SEND_IP_MOTOR_X)
    time.sleep(1)

f.close()
drive_motor(y_steps_per_step * y_images, 200, 0, 4, UDP_SEND_IP_MOTOR_Y)

print(x_measurements)



x = []
y = []
z = []

with open(logfile, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        y.append(float(row[0]))
        x.append(float(row[1]))
        z.append(float(row[2]))


print(x)
print(y)
print(z)

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
cntr1 = ax.contourf(xi, yi, zi, levels=20, cmap="gist_rainbow")

fig.colorbar(cntr1, ax=ax, orientation="horizontal").set_label("signal strength RSSI")
ax.plot(x, y, 'ko', ms=0.5)
ax.set(xlim=(latmin, latmax), ylim=(lngmin, lngmax))
ax.set_title("WiFi Map " + filetime)
ax.ticklabel_format(useOffset=False)

ax.set_aspect(1)
plt.savefig('plot_' + filetime + '.png', dpi=220)



#plt.plot(x_measurements)
#plt.ylabel('wifi strength')
#plt.show()

#exit()
#print(getSignal())
#print(getSignal())
#print(getSignal())
