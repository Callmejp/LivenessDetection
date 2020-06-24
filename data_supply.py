import os
from PIL import Image
import numpy as np

from Config import CHANNEL_SIZE, IMG_SIZE


def read_data(prefix_path, pos_num, neg_num):
    print("读取路径: " + prefix_path)
    sample_classification = ["pos_picture\\", "neg_picture\\"]
    result = []
    pos_pic_cnt = 0
    for index, classification in enumerate(sample_classification):
        if index == 0:
            flag = pos_num
        else:
            flag = neg_num

        for dir_num in range(1, flag):
            # e.g. train\\pos_picture\\1
            target_path = prefix_path + classification + str(dir_num)
            images = os.listdir(target_path)
            # print(len(images))
            number_of_images = len(images)

            buffer = []
            # Must in the `number` order to read image
            for image_index in range(number_of_images):
                # channel: CHANNEL_SIZE
                if len(buffer) >= CHANNEL_SIZE:
                    # temp = np.array(buffer)
                    # temp = temp.reshape(CHANNEL_SIZE, IMG_SIZE, IMG_SIZE, 1)
                    result.append(buffer)
                    buffer.pop(0)
                img = Image.open(target_path + "\\" + str(image_index) + ".bmp")
                # Normalization
                buffer.append(np.array(img, dtype=float) / 255.0)
        if index == 0:
            pos_pic_cnt = len(result)
    # ---------------------------------------------
    neg_pic_cnt = len(result) - pos_pic_cnt
    print("正样本: ", pos_pic_cnt, " ", "负样本: ", neg_pic_cnt)           
    batch_x = np.array(result, dtype=float)
    # print(batch_x.shape)
    y = []
    for i in range(pos_pic_cnt):
        y.append(np.array([1, 0], dtype=float))
    for i in range(neg_pic_cnt):
        y.append(np.array([0, 1], dtype=float))
    y = np.array(y)
    # y = np.expand_dims(y, axis=1)
    # print(y.shape)
    return batch_x, y

    # steps = 10
    # while True:
    #     for i in range(steps):
    #         yield batch_x[i: i+2], y[i: i+2]    


# my_gen()
 

