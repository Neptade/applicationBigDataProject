import os
from glob import glob
import time
import numpy as np
import pandas as pd
from tensorflow.image import resize
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

INPUT_PATH = os.getenv('INPUT_PATH', '../data')
MODEL_PATH = os.getenv('MODEL_PATH', '../model')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', '../output')

def load_data(img_paths):
    X = np.zeros(shape=(len(img_paths), 256,256,3))

    for i, path in enumerate(img_paths):
        X[i] = resize(img_to_array(load_img(path))/255., (256,256))

    return X

image_paths = sorted(glob(f'{INPUT_PATH}/*.jpg'))
images = load_data(image_paths)

class_names = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}
model_v3 = load_model(f'{MODEL_PATH}/ResNet152V2-Weather-Classification-03.h5')

preds = np.argmax(model_v3.predict(images), axis=-1)

labels = list(map(lambda x: class_names[x], list(preds)))

df = pd.DataFrame({'image_name': image_paths, 'prediction_label': labels})
timestamp = int(time.time() * 1000)
df.to_csv(f'{OUTPUT_PATH}/pred_{timestamp}.csv', index=False)
