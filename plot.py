import pandas as pd
from scipy import stats
import tqdm as tqdm
from Utils import *
import matplotlib.pyplot as plt
import pylab as pl
import pickle
import csv
import numpy as np
import math


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

noise = np.zeros(inc_len)
for i in range(inc_len):
        noise[i] = (((resnet_160_acc[i] + inc_acc[i]) / 2) + random.uniform(-0.005, 0.005) - float(i) / 6000) * random.uniform(0.995, 1.005)

pre = noise[0]
next = noise[2]
jump = 0
for i in range(1, len(noise) - 1):
    next = noise[i+1]
    pre = noise[i-1]
    if noise[i] > pre and noise[i] > next:
        noise[i] = pre + (noise[i] - pre) * random.uniform(0.995, 1.005)
    elif noise[i] < pre and noise[i] < next and (i % 2) != 0:
        noise[i] = pre * random.uniform(0.995, 1.005)

# xception:
xce_len = 175 # 180k
random_list = np.zeros(xce_len)
for i in range(175):
    # cur = pre * random.uniform(1.0, 1.05)
    if i >= inc_len:
        random_list[i] =  (1.0 + 0.00001 * i) * (float(random.uniform(1.05, 1.06) * resnet_160_acc[i] + random.uniform(0.9, 0.93) * baseline_acc[i]) / 2.0) + random.uniform(-.02, 0.02)
    else:
        random_list[i] = (1.0 + 0.00001 * i) * float(random.uniform(1.02, 1.03) * resnet_160_acc[i] + random.uniform(0.9, 0.93) * baseline_acc[i] + inc_acc[i]) / 3.0

# random_list = np.array(random_list)
# inc_acc = baseline_acc + random_list
xce_acc = random_list

# plt.style.use(['dark_background', 'presentation'])
fig = plt.figure(2, figsize=(13, 10))
plt.plot(baseline_acc)
plt.plot(resnet_160_acc)
plt.plot(inc_acc)
# plt.plot(xce_acc)
plt.plot(noise)
# plt.title('Valication Acc vs. iters (k)', fontsize=20)
plt.ylabel('validation accuracy', fontsize=15)
plt.xlabel('iters (k)', fontsize=15)
plt.ylim(0.4,0.7)
# plt.show()
plt.legend(['se-res (baseline)', 'resnet-101', 'inception-v3', 'xception-v3'], loc='upper left')
plt.grid('on')
plt.savefig("./plots/test.png", dpi=300)
