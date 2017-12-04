from __future__ import print_function

import os
from torch.autograd import Variable
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transform import *
from Utils import *
from cdimage import *
from torch.utils.data.sampler import RandomSampler
import operator
from tqdm import tqdm
# --------------------------------------------------------

from net.resnet101 import ResNet101 as Net

TTA_list = [random_shift_scale_rotate, random_crop]

use_cuda = True
IDENTIFIER = "resnet"
SEED = 123456
PROJECT_PATH = './project'
CDISCOUNT_HEIGHT = 180
CDISCOUNT_WIDTH = 180
CDISCOUNT_NUM_CLASSES = 5270

csv_dir = './data/'
root_dir = '../output/'
test_data_filename = 'test.csv'
validation_data_filename = 'validation.csv'

initial_checkpoint = "../checkpoint/" + IDENTIFIER + "/latest.pth"
res_path = "./test_res/" + IDENTIFIER + "_val_TTA.res"
validation_batch_size = 64

def ensemble_predict(cur_procuct_probs, num):
    candidates = np.argmax(cur_procuct_probs, axis=1)
    probs_means = np.mean(cur_procuct_probs, axis=0)
    winner_score = 0.0
    winner = None
    for candidate in candidates:
        # Adopt pre chosen criteria to abandan some instances
        candidate_score = probs_means[candidate] * num
        abandan_cnt = 0
        for probs in cur_procuct_probs:  # iterate each product instance
            if probs[candidate] < probs_means[candidate] - 0.2:
                # abandan this instance
                candidate_score -= probs[candidate]
                abandan_cnt += 1

        if candidate_score > winner_score:
            winner = candidate
            winner_score = candidate_score

    return winner

def TTA(images):
    images_TTA_list = []

    for transform in TTA_list:
        cur_images = []
        for image in images:
            cur_images.append(pytorch_image_to_tensor_transform(transform(image)))

        images_TTA_list.append(torch.stack(cur_images))

    return images_TTA_list

def evaluate_sequential_ensemble(net, loader, path):
    product_to_prediction_map = {}
    cur_procuct_probs = np.array([]).reshape(0,CDISCOUNT_NUM_CLASSES)
    cur_product_id = None
    transform_num = 1

    with open(path, "a") as file:
        file.write("_id,category_id\n")

        for iter, (images, image_ids) in enumerate(tqdm(loader), 0):
            image_ids = np.array(image_ids)

            # transforms
            images_list = TTA(images.numpy()) # a list of image batch using different transforms
            probs_list = []
            for images in images_list:
                images = Variable(images.type(torch.FloatTensor)).cuda()
                logits = net(images)
                probs  = ((F.softmax(logits)).cpu().data.numpy()).astype(float)
                probs_list.append(probs)
                # print(probs)

            start = 0
            end = 0
            for image_id in image_ids:
                product_id = imageid_to_productid(image_id)

                if cur_product_id == None:
                    cur_product_id = product_id

                if product_id != cur_product_id:
                    # a new product
                    # print("cur product: " + str(cur_product_id))

                    # find winner for previous product
                    num = (end - start) * transform_num # total number of instances for current product
                    ## get probs in range [start, end)
                    for probs in probs_list:
                        # print(probs)
                        cur_procuct_probs = np.concatenate((cur_procuct_probs, np.array(probs[start:end])), axis=0)

                    # do predictions
                    winner = ensemble_predict(cur_procuct_probs, num)
                    # print("winner: ", str(winner))

                    # save winner
                    product_to_prediction_map[cur_product_id] = winner

                    # update
                    start = end
                    cur_product_id = product_id
                    cur_procuct_probs = np.array([]).reshape(0,CDISCOUNT_NUM_CLASSES)

                end += 1

            np.concatenate((cur_procuct_probs, np.array(probs[start:end])), axis=0)

        # find winner for current product
        num = (end - start) * transform_num  # total number of instances for current product
        ## get probs in range [start, end)
        for probs in probs_list:
            np.concatenate((cur_procuct_probs, probs[start:end]), axis=0)

        # do predictions
        winner = ensemble_predict(cur_procuct_probs, num)

        # save winner
        product_to_prediction_map[cur_product_id] = winner

        for product_id, prediction in product_to_prediction_map.items():
            file.write(str(product_id) + "," + str(prediction) + "\n")

def write_test_result(path, product_to_prediction_map):
    with open(path, "a") as file:
        file.write("_id,category_id\n")

        for product_id, prediction in product_to_prediction_map.items():
            print(product_id)
            print(prediction)
            file.write(str(product_id) + "," + str(prediction) + "\n")

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    net = Net(in_shape = (3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    net.cuda()
    net.eval()

    if os.path.isfile(initial_checkpoint):
        print("=> loading checkpoint '{}'".format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        net.load_state_dict(checkpoint['state_dict'])  # load model weights from the checkpoint
        print("=> loaded checkpoint '{}'".format(initial_checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(initial_checkpoint))
        exit(0)

    dataset = CDiscountDataset(csv_dir + validation_data_filename, root_dir)
    loader  = DataLoader(
                        dataset,
                        sampler=SequentialSampler(dataset),
                        batch_size  = validation_batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = False)

    product_to_prediction_map = evaluate_sequential_ensemble(net, loader, res_path)

    print('\nsucess!')
