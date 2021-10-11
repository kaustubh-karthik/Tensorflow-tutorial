# from deep_utils import order_test_set, split_data
import csv
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.engine.training import Model
from deeplearning_models import streetsigns_model
from deep_utils import create_generators


if __name__ == '__main__':

    path_to_train = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/training_folder/train'
    path_to_val = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/training_folder/val'
    path_to_test = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/gtsrb/Test'
    batch_size = 64
    epochs = 100

    learn_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

    TRAIN = True
    TEST = True

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    num_classes = train_generator.num_classes
    print(train_generator)

    if TRAIN:
        path_to_save_models = './Models'
        checkpoint_saver = ModelCheckpoint(
            path_to_save_models,
            monitor = 'val_accuracy',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )

        early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 2)

        # model = streetsigns_model(num_classes)
        model = tf.keras.models.load_model('./Models')
        model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
        model.fit(train_generator, epochs = epochs, batch_size = batch_size, validation_data = val_generator, callbacks = [checkpoint_saver, early_stop])
        model.evaluate(test_generator, batch_size = batch_size)



        '''path_to_images = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/gtsrb/Test'
        path_to_csv = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/gtsrb/Test.csv'
        order_test_set(path_to_images, path_to_csv)'''

        '''path_to_data = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/gtsrb/Train'
        path_to_save_train = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/training_folder/train'
        path_to_save_val = '/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/training_folder/val'

        split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1)'''
    
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()
        
        print('-----EVALUATING VALIDATION SET-----')
        model.evaluate(val_generator)
        print('-----EVALUATING TEST SET-----')
        model.evaluate(test_generator)

