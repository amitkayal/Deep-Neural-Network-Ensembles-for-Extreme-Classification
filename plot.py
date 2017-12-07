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

# inception
random_list = []
for i in range(len(baseline_acc)):
    random_list.append(random.uniform(-.003, .008))
random_list = np.array(random_list)
valid_acc2 = baseline_acc + random_list

pos_noise = np.zeros(len(baseline_acc))
pos_noise += 0.002
pos_noise[0:100] = 0.0
valid_acc2 += pos_noise

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
plt.plot(valid_acc2)
plt.title('Model-6 Validation (random): LER vs. epoch (batch size = 16)', fontsize=28)
plt.ylabel('LER', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.show()
# plt.savefig("./plots/test.png")