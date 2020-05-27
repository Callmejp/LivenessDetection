import tensorflow as tf
from tensorflow import keras

from data import train_data, test_data
from model import get_model




if __name__ == "__main__":
    x, y = train_data()
    test_x, test_y = test_data()
    # print(y)
    model = get_model()
    model.compile(loss='binary_crossentropy',  optimizer='adam', metrics = ['accuracy'])
    model.fit(x, y, epochs=30, verbose=1, validation_data = (test_x, test_y))
    # model.fit(x, y, epochs=30, verbose=1, batch_size=2)
    print(model.predict(test_x))