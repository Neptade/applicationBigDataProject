# import libraries
import os
from glob import glob
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.image import resize
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
from PIL import Image, UnidentifiedImageError


# environment variables
INPUT_PATH = os.getenv('INPUT_PATH', './data')
MODEL_PATH = os.getenv('MODEL_PATH', './model')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')

USE_CACHE = os.getenv('USE_CACHE', 'False').lower() == 'true'

print('Running WITH cache' if USE_CACHE else 'Running WITHOUT cache')


# load the dataframe if it exists in the output file or create a new one
image_paths = sorted(glob(f'{INPUT_PATH}/*'))
if USE_CACHE and len(sorted(glob(f'{OUTPUT_PATH}/*.csv'))) != 0:
    df = pd.read_csv(sorted(glob(f'{OUTPUT_PATH}/*.csv'))[-1])
    image_paths_already_predicted = df['image_name'].tolist()
else:
    df = pd.DataFrame()
    image_paths_already_predicted = []

valid_image_extensions = {".jpg", ".jpeg", ".png"}
class_names = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}


# function to check if the image is valid
# return : a boolean
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


# function to load each image and resize them
# return : a numpy array
def load_image(img_path):
    img = load_img(img_path)
    img_array = img_to_array(img) / 255.0
    img_array = resize(img_array, (256, 256))
    return img_array


# function to check if the image is in the dataset
# return : a boolean and the path of the image
def is_in_dataset(img_path, images):
    img = load_image(img_path)

    for im_path in images:
        try:
            im = load_image(im_path)
            mse = np.mean((im - img)**2)
            if mse < 0.01:
                return (True, im_path)
        except (UnidentifiedImageError, FileNotFoundError, IOError):
            continue
    return (False, None)

# function to load all images
# return : a numpy array
def load_data(img_paths):
    X = np.zeros(shape=(len(img_paths), 256, 256, 3), dtype=np.float32)

    for i, path in enumerate(img_paths):
        X[i] = load_image(path)

    return X


# check if the data folder is empty
if len(image_paths) != 0:
    images_to_predict = []
    path_images_to_predict = []

    # check each image in the data folder
    for img_path in image_paths:
        if not is_valid_image(img_path):
            print(f"Error: {img_path} is not a valid image")
            continue
        
        # check if the image is already in the dataset
        already_exists, at_path = is_in_dataset(img_path, image_paths_already_predicted)
        if USE_CACHE and already_exists:
            print(f'{os.path.basename(img_path)} is already in the dataset as {os.path.basename(at_path)}')
        else:
            print(f'{os.path.basename(img_path)} is not in the dataset')
            path_images_to_predict.append(img_path)

    # check if there are images to predict
    if len(path_images_to_predict) > 0:
        
        # load the images to predict
        images_to_predict = load_data(path_images_to_predict)

        try:
            # load the model and run the predictions
            model_v3 = load_model(f'{MODEL_PATH}/ResNet152V2-Weather-Classification-03.h5')
            preds = np.argmax(model_v3.predict(images_to_predict), axis=-1)
            labels = list(map(lambda x: class_names[x], list(preds)))

            # create a dataframe with the predictions
            df = pd.concat([df, pd.DataFrame({'image_name': path_images_to_predict, 'prediction_label': labels})], ignore_index=True)

            # save the predictions in a csv file
            now = datetime.now()
            timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
            df.to_csv(f'{OUTPUT_PATH}/pred_{timestamp}.csv', index=False)
        except:
            print("Error: Could not load the model or run predictions")
    else:
        print("No new images to predict")
else:
    print("The data folder is empty")
