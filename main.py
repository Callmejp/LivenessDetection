import tensorflow as tf
from tensorflow import keras

from data_supply import read_data
from model import get_model




if __name__ == "__main__":
    x, y = read_data("train\\")
    test_x, test_y = read_data("test\\")
    # print(y)
    model = get_model()
    model.compile(loss='binary_crossentropy',  optimizer='adam', metrics = ['accuracy'])
    model.fit(x, y, batch_size=50, epochs=30, verbose=1, validation_data = (test_x, test_y))
    # model.fit(x, y, epochs=30, verbose=1, batch_size=2)
    # print(model.predict(test_x))