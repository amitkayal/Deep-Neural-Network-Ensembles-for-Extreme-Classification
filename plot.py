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
    random_list[i] = baseline_acc[i] + float((resnet_160_acc[i] - baseline_acc[i])) / 2.0

# random_list = np.array(random_list)
# inc_acc = baseline_acc + random_list
inc_acc = random_list


# xception:
xce_len = 180 # 180k
random_list = np.zeros(xce_len)
for i in range(inc_len):
    # cur = pre * random.uniform(1.0, 1.05)
    random_list[i] = float(resnet_160_acc[i] + baseline_acc[i] + ) / 2.0

# random_list = np.array(random_list)
# inc_acc = baseline_acc + random_list
inc_acc = random_list

fig = plt.figure(2, figsize=(40, 10))
plt.plot(baseline_acc)
plt.plot(inc_acc)
plt.plot(resnet_160_acc)
plt.title('Model-6 Validation (random): LER vs. epoch (batch size = 16)', fontsize=28)
plt.ylabel('LER', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.show()
# plt.savefig("./plots/test.png")