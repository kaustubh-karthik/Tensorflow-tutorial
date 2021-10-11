import tensorflow, numpy as np
from deeplearning_models import functional_model, inherit_model, sequential_model
from deep_utils import display_examples

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    # display_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis = -1)
    x_test = np.expand_dims(x_test, axis = -1)

    '''MODELS'''

    # model = sequential_model # Sequential model
    # model = functional_model() # Functional model
    model = inherit_model() # Class model

    '''RUNNING'''
    # Compiling the model
    model.compile(optimizer = 'adam', loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(), metrics = 'accuracy')

    # Model training
    model.fit(x_train, y_train, batch_size = 32, epochs = 1)

    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size = 64)

    
