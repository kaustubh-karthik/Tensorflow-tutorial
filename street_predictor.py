import tensorflow as tf
import numpy as np
import glob


def predict_model(model, img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels = 3)
    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    image = tf.image.resize(image, [51, 50])
    image = tf.expand_dims(image, axis = 0)

    prediction = model.predict(image)
    prediction = np.argmax(prediction)

    return prediction

if __name__ == '__main__':

    predictions = []
    # Test folder
    '''for image in glob.glob('/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/gtsrb/Test/14/*.png'):
        model = tf.keras.models.load_model('./Models')

        prediction = predict_model(model, image)
        predictions.append(prediction)
        
        print(f'prediction: {prediction}')'''

    model_select = int(input('Enter model path: '))
    models = {x: model for x, model in enumerate(glob.glob('/Users/kaustubhkarthik/Documents/Tensorflow-tutorial/*model'))}

    print(models)

    model = tf.keras.models.load_model(models[model_select])
    image = input('Enter image path: ')

    prediction = predict_model(model, image)
    predictions.append(prediction)
    
    print(f'prediction: {prediction}')
