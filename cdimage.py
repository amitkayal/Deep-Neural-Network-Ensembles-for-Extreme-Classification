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
    def __init__(self, csv_file, root_dir, transform=None):
        self.train_names=[]
        self.root_dir=root_dir
        self.transform = transform
        train_images = pd.read_csv(csv_file)
        train_ids = list(train_images['product_id'])
        train_idxs = list(train_images['image_id'])
        self.labels = list(train_images['category_id'])
        num_train = len(train_images)
        print(num_train)
        for i in range(num_train):
            train_name = '{}/{}-{}.jpg'.format(self.labels[i],train_ids[i],train_idxs[i])
            self.train_names.append(train_name)
        print(self.train_names)

    def __len__(self):
        return len(self.train_names)

    def __getitem__(self, idx):
        img = cv2.imread(self.root_dir + self.train_names[idx])
        plt.imshow(img)
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img,label




def pytorch_image_to_tensor_transform(image):

    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor

def pytorch_tensor_to_image_transform(tensor):
    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]

    tensor[0] = tensor[0]*std[0] + mean[0]
    tensor[1] = tensor[1]*std[1] + mean[1]
    tensor[2] = tensor[2]*std[2] + mean[2]


    image = tensor.numpy()*255
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return image


def image_to_tensor_transform(image):
    tensor = pytorch_image_to_tensor_transform(image)
    tensor[0] = tensor[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    tensor[1] = tensor[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    tensor[2] = tensor[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    return tensor


def train_augment(image):
    if random.random() < 0.5:
        image = random_shift_scale_rotate(image,
                                          # shift_limit  = [0, 0],
                                          shift_limit=[-0.06, 0.06],
                                          scale_limit=[0.9, 1.2],
                                          rotate_limit=[-10, 10],
                                          aspect_limit=[1, 1],
                                          # size=[1,299],
                                          borderMode=cv2.BORDER_REFLECT_101, u=1)
    else:
        pass

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)

    tensor = image_to_tensor_transform(image)
    return tensor
# dataset = CDiscountDataset(csv_dir+data_file_name,root_dir,transform=transforms.ToTensor())
# sampler = SequentialSampler(dataset)
#
#
# data_loader= DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0)
# for i, (images, indices) in enumerate(data_loader, 0):
#
#     batch_size = len(indices)
#     num_augments = len(images)
#     print('batch_size = %d' % batch_size)
#     print('num_augments = %d' % num_augments)
#     for a in range(num_augments):
#         tensor = images[a][0]
#         print('%d: %s' % (a, str(tensor.size())))
#         image = pytorch_tensor_to_image_transform(tensor)
#         plt.imshow('image%d' % a, image)
#     cv2.waitKey(0)
#     xx = 0

# for iter, (images, labels) in enumerate(data_loader):
#     print(type(images))
#     #print(images)
#     plt.imshow(images[0].numpy())
#     plt.show()
#     break
