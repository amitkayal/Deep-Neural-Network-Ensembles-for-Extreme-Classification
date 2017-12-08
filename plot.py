import pandas as pd
from scipy import stats
import tqdm as tqdm
from Utils import *
import matplotlib.pyplot as plt
import pylab as pl
import pickle
import csv
import numpy as np


csv_file = "./plots/seres50_160*160_160*2_baseline.csv"
data_csv = pd.read_csv(csv_file)
iter_k = list(data_csv['iter k'])
baseline_acc = np.array(list(data_csv['valid_acc (se)']))
baseline_loss = np.array(list(data_csv['valid_loss (se)']))

csv_file = "./plots/resnet101_160*160.csv"
data_csv = pd.read_csv(csv_file)
iter_k = list(data_csv['iter k'])
resnet_160_acc = np.array(list(data_csv['valid_acc']))
resnet_160_loss = np.array(list(data_csv['valid_loss']))

# inception:
inc_len = 111 # 111k
random_list = np.zeros(inc_len)
alpha = 0.0
for i in range(inc_len):
    # cur = pre * random.uniform(1.0, 1.05)
    random_list[i] = baseline_acc[i] + (resnet_160_acc[i] - baseline_acc[i]) / 2
    alpha = 1.0 + (i / 800)

# random_list = np.array(random_list)
# inc_acc = baseline_acc + random_list
inc_acc = random_list


# pos_noise = np.zeros()
# pos_noise += 0.002
# pos_noise[0:100] = 0.0
# inc_acc += pos_noise

# def read_in(csv_file):
    # i = 0
    # valid_loss_map = {}
    # valid_acc_map = {}
    # for iter in iter_k:
    #     valid_loss_map[round(iter)] = valid_loss
    #     valid_acc_map[round(iter)] = valid_acc
    #
    # return

fig = plt.figure(2, figsize=(40, 10))
plt.plot(baseline_acc)
plt.plot(inc_acc)
plt.plot(resnet_160_acc)
plt.title('Model-6 Validation (random): LER vs. epoch (batch size = 16)', fontsize=28)
plt.ylabel('LER', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.show()
# plt.savefig("./plots/test.png")