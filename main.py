import tensorflow as tf
from tensorflow import keras

from data_supply import read_data
from model import get_model




if __name__ == "__main__":
    x, y = read_data("train\\", 15, 15)
    test_x, test_y = read_data("test\\", 7, 7)
    # print(y)
    model = get_model()
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics = ['accuracy'])
    model.fit(x, y, batch_size=2, epochs=1, verbose=1, validation_data = (test_x, test_y))
    # model.fit(x, y, epochs=30, verbose=1, batch_size=2)
    # print(model.predict(test_x))