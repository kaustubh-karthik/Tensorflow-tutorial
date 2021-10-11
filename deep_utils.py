import matplotlib as plt, numpy as np, os, glob, shutil, csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def display_examples(examples, labels):

    plt.figure(figsize=(10, 10))

    for i in range (25):

        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.imshow(img, cmap = 'gray')
        plt.tight_layout()

    plt.show()

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size = 0.1):

    folders = next(os.walk(path_to_data))[1]

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))

        # print('-----', full_path, images_paths, folder, folders, sep = '-----')

        x_train, x_val = train_test_split(images_paths, test_size = split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)
        for x in x_val:
        
            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

def order_test_set(path_to_images, path_to_csv):

    testset = {}

    try:
        with open(path_to_csv, 'r') as csv_file:

            reader = csv.reader(csv_file, delimiter = ',')

            for i, row in enumerate(reader):
                if i == 0:
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('ERROR READING CSV FILE')

def create_generators(batch_size, train_data_path,  test_data_path, val_data_path = None):
    
    preprocessor = ImageDataGenerator(rescale = 1 / 255)

    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode = 'categorical',
        target_size = (51, 50),
        color_mode = 'rgb',
        shuffle = True,
        batch_size = batch_size
    )

    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode = 'categorical',
        target_size = (51, 50),
        color_mode = 'rgb',
        shuffle = False,
        batch_size = batch_size
    )

    if val_data_path:
        val_generator = preprocessor.flow_from_directory(
            val_data_path,
            class_mode = 'categorical',
            target_size = (51, 50),
            color_mode = 'rgb',
            shuffle = False,
            batch_size = batch_size
        )

        return train_generator, test_generator, val_generator

    else:        
        return train_generator, test_generator

def create_save_stop(save_path, save_monitor, mode = 'max', stop_monitor = 'accuracy', patience = 2):
    
    checkpoint_saver = ModelCheckpoint(
    save_path,
    monitor = save_monitor,
    mode = mode,
    save_best_only = True,
    save_freq = 'epoch',
    verbose = 1
)
    early_stop = EarlyStopping(monitor = stop_monitor, patience = patience)

    return checkpoint_saver, early_stop

