from __future__ import print_function
import os
import cv2
import numpy as np
import pickle
import keras
import variables as vars
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Config
img_rows, img_cols = vars.img_rows, vars.img_cols
batch_size = vars.batch_size
num_classes = vars.num_classes
epochs = vars.epochs
model_json_path = vars.model_json_path
model_path = vars.model_path
prediction_file_dir_path = vars.prediction_file_dir_path

data = []
labels = []
path = 'FEATURE-BASED-IMAGES/'

for folder, subfolders, files in os.walk(path):
    for name in files:
        if name.endswith('.jpg'):
            img_path = os.path.join(folder, name)
            x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.resize(x, (img_rows, img_cols))
            _, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

            # Morphological dilation
            struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cpy = cv2.dilate(~x, struct, iterations=1)
            x = ~cpy

            x = np.expand_dims(x, axis=-1)  # Add channel dimension
            data.append(x)
            labels.append(os.path.basename(folder))

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=0)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

input_shape = (img_rows, img_cols, 1)

# Build model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

# Encode labels
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Train
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
with open(model_json_path, "w") as json_file:
    json_file.write(model.to_json())
model.save_weights(model_path)
with open(vars.label_obj_path, 'wb') as lb_obj:
    pickle.dump(lb, lb_obj)

print("Saved model to disk")