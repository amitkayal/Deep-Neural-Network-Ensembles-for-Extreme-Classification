from __future__ import print_function

import os
from datetime import *
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from timeit import default_timer as timer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

# custom modules
from Log import *
from StepLR import *
from Utils import *
from AverageMeter import *
from cdimage import *

out_dir  = '../'
csv_dir = './data/'
root_dir = '../output/'
validation_data_filename = 'validation_small.csv'

train_dataset = CDiscountDemoDataset(csv_dir + validation_data_filename, root_dir, transform=None)
train_loader = DataLoader(
    train_dataset,
    sampler=SequentialSampler(train_dataset),
    batch_size=32,
    drop_last=False,
    num_workers=1,
    pin_memory=False)

train_loader_iter = iter(train_loader)
for i in range(8):
    print("-------------------- %d --------------------\n"%i)
    cur = next(train_loader_iter, None)
    if cur == None:
        print("restart")
        train_loader_iter = iter(train_loader)
        continue
    cnt = cur[0]
    idx = cur[1]
    print("cnt: ", cnt, ", idx: ", idx)

# for (cnt, idx) in train_loader:
#     print("cnt: ", cnt, ", idx: ", idx)
#     break
#
# for (cnt, idx) in train_loader:
#     print("cnt: ", cnt, ", idx: ", idx)
#     break