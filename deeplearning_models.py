import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Model


sequential_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28, 28, 1)), 
        Conv2D(32, (3, 3), activation = 'relu'),
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128 ,(3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax')
    ]
)

# functional approach : function that returns a model
def functional_model():
    shape_in = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation = 'relu')(shape_in)
    x = Conv2D(64, (3, 3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128 ,(3, 3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(10, activation = 'softmax')(x)

    model = tensorflow.keras.Model(inputs = shape_in, outputs = x)

    return model

# tensorflow.keras.Model - inheritance and overriding of class methods
class inherit_model(tensorflow.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32, (3, 3), activation = 'relu')
        self.conv2 = Conv2D(64, (3, 3), activation = 'relu')
        self.max_pool1 = MaxPool2D()
        self.batch_normal1 = BatchNormalization()

        self.conv3 = Conv2D(128 ,(3, 3),activation = 'relu')
        self.max_pool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAveragePooling2D()
        self.dense1 = Dense(64, activation = 'relu')
        self.dense2 = Dense(10, activation = 'softmax')

    def call(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.batch_normal1(x)
        x = self.conv3(x)
        x = self.max_pool1(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def streetsigns_model():

    inp = Input(shape=(51, 50, 3))

    x = Conv2D(64, (3, 3), activation = 'relu')(inp)
    x = Conv2D(64, (3, 3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128 ,(3, 3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(43, activation = 'softmax')(x)

    model = tensorflow.keras.Model(inputs = inp, outputs = x)

    return model

def fish_model():

    # Causes 'requires input with multiple of x'
    inp = Input(shape = (51, 50, 3))

    x = Conv2D(64, (3, 3), activation = 'relu')(inp)
    x = Conv2D(64, (3, 3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128 ,(3, 3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(9, activation = 'softmax')(x)

    model = tensorflow.keras.Model(inputs = inp, outputs = x)

    return model
    