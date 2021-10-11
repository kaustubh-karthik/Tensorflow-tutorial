from deeplearning_models import fish_model
from deep_utils import create_generators
import tensorflow as tf
from deep_utils import create_save_stop


train_path = '/Users/kaustubhkarthik/Downloads/fish_archive/Fish_Dataset/Fish_Dataset/full_picture'
test_path = '/Users/kaustubhkarthik/Downloads/fish_archive/NA_Fish_Dataset'

train_gen, test_gen = create_generators(64, train_path, test_path)

save_path = './Fish_model'

# model = fish_model()

model = tf.keras.models.load_model('./Fish_model')

model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
model.fit(train_gen, epochs = 100, batch_size = 64, callbacks = [create_save_stop(save_path, 'accuracy')])
model.evaluate(test_gen, batch_size = 64)
