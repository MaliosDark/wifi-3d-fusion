import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import yaml
import tqdm


datadir = "data/BLE/"

def fit_T_gamma(distance, RSSI):
    """RSSI = T - 10 * gamma * log10(d)
    """

    log_distance = np.log10(np.array(distance)).reshape(-1, 1)
    RSSI = np.array(RSSI).reshape(-1, 1)
    model = LinearRegression()
    model.fit(log_distance, RSSI)

    # The slope (m) and intercept (c) of the model:
    m = model.coef_[0][0]
    c = model.intercept_[0]
    T = c
    gamma = -m / 10

    return T, gamma




def load_data(data_index):

    tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
    gateway_pos_dir = os.path.join(datadir, 'gateway_position.yml')
    rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')


    # load gateway position
    with open(os.path.join(gateway_pos_dir)) as f:
        gateway_pos_dict = yaml.safe_load(f)
        gateway_pos = np.array([pos for pos in gateway_pos_dict.values()], dtype=np.float32)

    # Load transmitter position
    tx_poses = pd.read_csv(tx_pos_dir).values

    # Load gateway received RSSI
    rssis = pd.read_csv(rssi_dir).values

    rssi_all, dis_all = [], []
    for idx in data_index:
        rssi_irow = rssis[idx]
        tx_pos = tx_poses[idx].flatten()  # [3]
        for i_gateway, rssi in enumerate(rssi_irow):
            if rssi != -100:
                dis = np.linalg.norm(tx_pos - gateway_pos[i_gateway])
                rssi_all.append(rssi)
                dis_all.append(dis)

    return np.array(dis_all), np.array(rssi_all)



def mri():

    train_index = np.loadtxt(os.path.join(datadir, 'train_index.txt'), dtype=int)
    test_index = np.loadtxt(os.path.join(datadir, 'test_index.txt'), dtype=int)
    train_dis, train_rssi = load_data(train_index)
    test_dis, test_rssi = load_data(test_index)

    T, gamma = fit_T_gamma(train_dis, train_rssi)

    errors = []
    for i in range(len(test_dis)):
        dis = test_dis[i]
        rssi = test_rssi[i]
        rssi_pred = T - 10 * gamma * np.log10(dis)
        # print(rssi, rssi_pred)
        error = abs(rssi - rssi_pred)
        errors.append(error)
    print("test median error: {:.2f} dB".format(np.median(errors)))
    print(f"fitting T: {T}, gamma: {gamma}")




if __name__ == '__main__':

    mri()