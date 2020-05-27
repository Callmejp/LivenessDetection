import os
from PIL import Image
import numpy as np


def train_data():
    pos_path = "train\\pos_picture\\"
    neg_path = "train\\neg_picture\\"
    result = []
    for dir_num in range(1, 11):
        pos_images = os.listdir(pos_path + str(dir_num))
        neg_images = os.listdir(neg_path + str(dir_num))

        pos = []
        for pos_image in pos_images:
            img = Image.open(pos_path + str(dir_num) + "\\" + pos_image)
            pos.append(np.array(img, dtype=float))
            
        neg = []
        for neg_image in neg_images:
            img = Image.open(neg_path + str(dir_num) + "\\" + neg_image)
            neg.append(np.array(img, dtype=float))
    
        result.append(pos)
        result.append(neg)
    
    batch_x = np.array(result, dtype=float)
    # print(batch_x.shape)
    y = []
    for i in range(20):
        if i % 2 == 0:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    y = np.expand_dims(y, axis=1)
    # print(y.shape)
    return batch_x, y

    # steps = 10
    # while True:
    #     for i in range(steps):
    #         yield batch_x[i: i+2], y[i: i+2]    


# my_gen()

def test_data():
    pos_path = "test\\pos_picture\\"
    neg_path = "test\\neg_picture\\"
    result = []
    for dir_num in range(1, 11):
        pos_images = os.listdir(pos_path + str(dir_num))
        neg_images = os.listdir(neg_path + str(dir_num))

        pos = []
        for pos_image in pos_images:
            img = Image.open(pos_path + str(dir_num) + "\\" + pos_image)
            pos.append(np.array(img, dtype=float))
            
        neg = []
        for neg_image in neg_images:
            img = Image.open(neg_path + str(dir_num) + "\\" + neg_image)
            neg.append(np.array(img, dtype=float))
    
        result.append(pos)
        result.append(neg)
    
    batch_x = np.array(result, dtype=float)
    # print(batch_x.shape)
    y = []
    for i in range(20):
        if i % 2 == 0:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    y = np.expand_dims(y, axis=1)
    # print(y.shape)
    return batch_x, y

 

