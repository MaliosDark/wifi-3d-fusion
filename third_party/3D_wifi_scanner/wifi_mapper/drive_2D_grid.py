import sys
import json
import requests
import time
import socket
import matplotlib.pyplot as plt
import csv

f = open('csv_file_' + str(time.time()) + '.csv', 'w')
writer = csv.writer(f)

UDP_SEND_IP_MOTOR_Y = "192.168.1.177"
UDP_SEND_IP_MOTOR_X = "192.168.1.178"
UDP_SEND_PORT = 8888

url = 'http://192.168.1.103/rssi'

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
    return data['signal']


x_images = 24
y_images = 20
x_steps_per_step = 400
y_steps_per_step = 1000

drive_motor(x_steps_per_step * x_images, 100, 0, 8, UDP_SEND_IP_MOTOR_X)

drive_motor(y_steps_per_step * y_images, 200, 1, 4, UDP_SEND_IP_MOTOR_Y)
input("Press Enter to continue...")
drive_motor(x_steps_per_step * x_images, 100, 1, 8, UDP_SEND_IP_MOTOR_X)
drive_motor(y_steps_per_step * y_images, 200, 0, 4, UDP_SEND_IP_MOTOR_Y)
input("Press Enter to start scan...")

for y in range(y_images):
    drive_motor(y_steps_per_step, 200, 1, 4, UDP_SEND_IP_MOTOR_Y)
    time.sleep(1.5)

    for x in range(x_images):
        drive_motor(x_steps_per_step, 100, 0, 8, UDP_SEND_IP_MOTOR_X)
        time.sleep(0.5)
        signal = getSignal()
        # x_measurements.append(signal)
        data = []
        data.append(y)
        data.append(x)
        data.append(signal)
        time.sleep(0.5)
        writer.writerow(data)

    drive_motor(x_steps_per_step * x_images, 100, 1, 8, UDP_SEND_IP_MOTOR_X)
    time.sleep(4)

f.close()
drive_motor(y_steps_per_step * y_images, 200, 0, 4, UDP_SEND_IP_MOTOR_Y)

print(x_measurements)

plt.plot(x_measurements)
plt.ylabel('wifi strength')
plt.show()

exit()
print(getSignal())
print(getSignal())
print(getSignal())
