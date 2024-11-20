import os
from glob import glob
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.image import resize
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
from PIL import Image, UnidentifiedImageError

INPUT_PATH = os.getenv('INPUT_PATH', './data')
MODEL_PATH = os.getenv('MODEL_PATH', './model')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')
valid_image_extensions = {".jpg", ".jpeg", ".png"}

def is_valid_image(file_path):
    if os.path.splitext(img_path)[1].lower() in valid_image_extensions:
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, FileNotFoundError, IOError):
            return False
    else:
        return False

def load_data(img_paths):
    X = np.zeros(shape=(len(img_paths), 256,256,3))

    for i, path in enumerate(img_paths):
        X[i] = resize(img_to_array(load_img(path))/255., (256,256))

    return X

image_paths = sorted(glob(f'{INPUT_PATH}/*'))
if(len(image_paths)!=0):
    path_images_to_predict = []

    for img_path in image_paths:
        if not is_valid_image(img_path):
            print(f"Error: {img_path} is not a valid image")
            continue
        path_images_to_predict.append(img_path)

    images = load_data(path_images_to_predict)

    class_names = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}
    model_v3 = load_model(f'{MODEL_PATH}/ResNet152V2-Weather-Classification-03.h5')

    preds = np.argmax(model_v3.predict(images), axis=-1)

    labels = list(map(lambda x: class_names[x], list(preds)))

    df = pd.DataFrame({'image_name': path_images_to_predict, 'prediction_label': labels})
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    df.to_csv(f'{OUTPUT_PATH}/pred_{timestamp}.csv', index=False)
else:
    print("The data folder is empty")