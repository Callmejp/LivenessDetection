import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D

from Config import IMG_SIZE, CHANNEL_SIZE


# def get_model():
#     model = Sequential() 
#     model.add(Conv2D(16, (4, 4), strides=(2,2), activation='relu', padding='same',
#     data_format='channels_first',name='layer1_con1',input_shape=(CHANNEL_SIZE, IMG_SIZE, IMG_SIZE)))
#     model.add(MaxPooling2D(pool_size=(4, 4),strides=(2,2), padding = 'same',
#     data_format='channels_first', name = 'layer1_pool'))

#     model.add(Conv2D(8, (4, 4), strides=(2,2), activation='relu', padding='same',
#     data_format='channels_first',name='layer2_con1'))
#     model.add(MaxPooling2D(pool_size=(4, 4),strides=(2,2), padding = 'same',
#     data_format='channels_first', name = 'layer2_pool'))

#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))  
#     # model.add(Dropout(0.5))
#     model.add(Dense(1, activation='softmax')) 
#     print(model.summary())
#     return model

def get_model():
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 3),
                    activation='relu',
                    input_shape=(CHANNEL_SIZE, IMG_SIZE, IMG_SIZE, 1)))
    model.add(Conv3D(32, (3, 3, 3), data_format='channels_first', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    return model