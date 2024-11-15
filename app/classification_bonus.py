import os
import numpy as np
import pandas as pd
import time
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model

MODEL_PATH = os.getenv('MODEL_PATH', './model')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')
INPUT_PATH = os.getenv('INPUT_PATH', './data')

image_paths = sorted(glob(f'{INPUT_PATH}/*.jpg'))
df = pd.read_csv(sorted(glob(f'{OUTPUT_PATH}/*.csv'))[-1])
image_paths_already_predicted = df['image_name'].tolist()

def is_in_dataset(img_path, images):
    img = load_image(img_path)

    for i, im_path in enumerate(images):
        im = load_image(im_path)
        mse = np.mean((im - img)**2)
        if mse < 0.01:
            return 1
    return 0

def load_image(img_path):
    img = load_img(img_path)
    img_array = img_to_array(img) / 255.0
    img_array = resize(img_array, (256, 256))
    return img_array

def load_data(img_paths):
    X = np.zeros(shape=(len(img_paths), 256, 256, 3), dtype=np.float32)

    for i, path in enumerate(img_paths):
        img_array = load_image(path)
        X[i] = img_array

    return X

images_to_predict = []
path_images_to_predict = []

for img_path in image_paths:
    if is_in_dataset(img_path, image_paths_already_predicted):
        print(f'{img_path} is already in the dataset')
    else:
        print(f'{img_path} is not in the dataset')
        path_images_to_predict.append(img_path)

if path_images_to_predict:
    images_to_predict = load_data(path_images_to_predict)

    class_names = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}
    model_v3 = load_model(f'{MODEL_PATH}/ResNet152V2-Weather-Classification-03.h5')

    preds = np.argmax(model_v3.predict(images_to_predict), axis=-1)

    labels = list(map(lambda x: class_names[x], list(preds)))

    df = pd.concat([df, pd.DataFrame({'image_name': path_images_to_predict, 'prediction_label': labels})], ignore_index=True)
    
    timestamp = int(time.time() * 1000)

    df.to_csv(f'{OUTPUT_PATH}/pred_{timestamp}.csv', index=False)
else:
    print("No new images to predict.")
