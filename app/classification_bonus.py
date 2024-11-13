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

print(image_paths_already_predicted)
print(image_paths)

images_to_predict = []
path_images_to_predict = []

for img_path in image_paths:
    if is_in_dataset(img_path, image_paths_already_predicted):
        print(f'{img_path} is already in the dataset')
    else:
        print(f'{img_path} is not in the dataset')
        path_images_to_predict.append(img_path)

print(path_images_to_predict)