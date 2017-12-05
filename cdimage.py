from label_category_transform import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transform import *
import torch
import random

csv_dir = './data/'
root_dir = '../output/train/'
data_file_name = 'train_data.csv'


class CDiscountDataset(Dataset):
    def __init__(self, csv_dir, root_dir, transform=None):
        # print("loading CDiscount Dataset...")
        self.image_names=[]
        self.root_dir=root_dir
        self.transform = transform
        image_data = pd.read_csv(csv_dir)
        #train_ids = list(train_images['product_id'])
        self.image_id = list(image_data['image_id'])
        self.labels = list(image_data['category_id'])
        self.indexes = list(image_data['category_id'])
        num_train = len(image_data)
        # print(num_train)
        # print("dataset labels",self.labels)
        for i in range(num_train):
            self.indexes[i] = category_id_to_index[self.labels[i]]
            image_name = '{}/{}.jpg'.format(self.labels[i],self.image_id[i])
            self.image_names.append(image_name)
        # print("label type:",type(self.labels))
        # print("label size:",len(self.labels))
        # print("label content:",self.labels[0:10])
        #print(self.train_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = cv2.imread(self.root_dir + 'train/'+ self.image_names[idx])
        label = self.indexes[idx]
        img_id = self.image_id[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_id


class CDiscountValidDataset(Dataset):
    def __init__(self, csv_dir, root_dir, transform=None):
        # print("loading CDiscount Dataset...")
        self.image_names=[]
        self.root_dir=root_dir
        self.transform = transform
        image_data = pd.read_csv(csv_dir)
        #train_ids = list(train_images['product_id'])
        self.image_id = list(image_data['image_id'])
        self.labels = list(image_data['category_id'])
        self.indexes = list(image_data['category_id'])
        num_train = len(image_data)
        # print(num_train)
        # print("dataset labels",self.labels)
        for i in range(num_train):
            self.indexes[i] = category_id_to_index[self.labels[i]]
            image_name = '{}/{}.jpg'.format(self.labels[i],self.image_id[i])
            self.image_names.append(image_name)
        # print("label type:",type(self.labels))
        # print("label size:",len(self.labels))
        # print("label content:",self.labels[0:10])
        #print(self.train_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = cv2.imread(self.root_dir + 'train/'+ self.image_names[idx])
        label = self.indexes[idx]
        img_id = self.image_id[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_id

class CDiscountTestDataset(Dataset):
    def __init__(self, csv_dir, root_dir, transform=None):
        # print("loading CDiscount Dataset...")
        self.image_names=[]
        self.root_dir=root_dir
        self.transform = transform
        image_data = pd.read_csv(csv_dir)
        self.image_id = list(image_data['image_id'])
        self.labels = list(image_data['category_id'])
        self.indexes = list(image_data['category_id'])
        num_train = len(image_data)
        # print(num_train)
        # print("dataset labels",self.labels)
        for i in range(num_train):
            self.indexes[i] = category_id_to_index[self.labels[i]]
            image_name = '{}.jpg'.format(self.image_id[i])
            self.image_names.append(image_name)
        # print("label type:",type(self.labels))
        # print("label size:",len(self.labels))
        # print("label content:",self.labels[0:10])
        #print(self.train_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = cv2.imread(self.root_dir + 'test/'+ self.image_names[idx])
        label = self.indexes[idx]
        image_id = self.image_id[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, image_id

class CDiscountDemoDataset(Dataset):
    def __init__(self, csv_dir, root_dir, transform=None):
        print("loading CDiscount Dataset...")

        self.cnt = 0

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        self.cnt += 1
        return self.cnt, idx









