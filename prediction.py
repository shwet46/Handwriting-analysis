from __future__ import print_function
# coding: utf-8

import os
import cv2
import keras
import pickle
import numpy as np
import variables as vars
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

img_rows, img_cols = vars.img_rows, vars.img_cols
batch_size = vars.batch_size
num_classes = vars.num_classes
epochs = vars.epochs
model_json_path = vars.model_json_path
model_path = vars.model_path
prediction_file_dir_path = vars.prediction_file_dir_path
label_obj_path = vars.label_obj_path

# ---------------------------- Prediction Utils ----------------------------

def print_results(class_lbl, out):
    print('\n' + '~' * 60)
    for k, lbl in enumerate(class_lbl):
        confidence = out[k] * 100
        if lbl == 'LEFT_MARG':
            print('\n > Courageous :', '\t' * 5, f"{confidence:.2f}%")
            print('\n > Insecure and devotes oneself completely :\t',
                  f"{100 - confidence:.2f}%")
        elif lbl == 'RIGHT_MARG':
            print('\n > Avoids future and a reserved person :\t', f"{confidence:.2f}%")
        elif lbl == 'SLANT_ASC':
            print('\n > Optimistic :', '\t' * 5, f"{confidence:.2f}%")
        elif lbl == 'SLANT_DESC':
            print('\n > Pessimistic :', '\t' * 4, f"{confidence:.2f}%")
    print('~' * 60 + '\n')


def predict_personalities(filename):
    try:
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_path)
        print("***** Loaded Model from disk *****")
    except Exception as e:
        return f'\n\n> Need to train the model first!\nError: {e}'

    img_path = os.path.join(prediction_file_dir_path, filename)
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (img_rows, img_cols))
    _, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

    # Morphological dilation
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cpy = cv2.dilate(~x, struct, iterations=1)
    x = ~cpy

    x = np.expand_dims(x, axis=-1)  # Add channel
    x = np.expand_dims(x, axis=0)   # Add batch

    x = x.astype('float32') / 255.0  # Normalize

    out = loaded_model.predict(x, batch_size=32, verbose=0)

    with open(label_obj_path, 'rb') as lb_obj:
        lb = pickle.load(lb_obj)

    result = lb.inverse_transform([np.argmax(out[0])])
    print_results(lb.classes_, out[0])

    return f'\n> Prediction Completed! Result: {result[0]}'


if __name__ == '__main__':
    file_list = None
    for dirpath, _, files in os.walk(prediction_file_dir_path):
        file_list = files
        break

    if file_list:
        res = predict_personalities(file_list[0])
        print(res)
    else:
        print('No file found for prediction!')