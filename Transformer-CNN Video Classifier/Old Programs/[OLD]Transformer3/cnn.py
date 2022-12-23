import os
from turtle import pd
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import pandas

image_height, image_width = 64, 64
max_images_per_class = 1500

dataset_directory = "dataset/train"
classes_list = os.listdir('dataset/train')

model_output_size = len(classes_list)

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    while True:
        success, frame = video_reader.read() 
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

def create_dataset():
    temp_features = [] 
    features = []
    labels = []
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(dataset_directory, class_name))
        couuunt = 0
        for file_name in files_list:
            video_file_path = os.path.join(dataset_directory, class_name, file_name)
            frames = frames_extraction(video_file_path)
            temp_features.extend(frames)
            print(couuunt)
            couuunt+=1
        features.extend(random.sample(temp_features, max_images_per_class))
        print("4")
        labels.extend([class_index] * max_images_per_class)
        temp_features.clear()
    features = np.asarray(features)
    labels = np.array(labels)  
    return features, labels

#features, labels = create_dataset()
#np.save("mediumdata.npy", features)
#np.save("mediumlabels.npy", labels)
features = np.load("mediumdata.npy")
labels = np.load("mediumlabels.npy")
print("DATA LOADED")

one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.2, shuffle = True)
def create_model():
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(model_output_size, activation = 'softmax'))
    model.summary()
    return model
model = create_model()

print("Model Created Successfully!")
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ["accuracy"])
model_training_history = model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 32, shuffle = True, validation_split = 0.1, callbacks = [early_stopping_callback])
#model_evaluation_history = model.evaluate(features_test, labels_test)
model.save("hypermodel.h5")
np.save("modeled_labels.npy", np.array(labels_test))
np.save("modeled_data.npy", np.array(features_test))
preds = model.predict(features_test)
preds = np.argmax(preds, axis=-1)
labels_test = np.argmax(labels_test, axis=-1)
print(preds)
print(labels_test)
from sklearn.metrics import confusion_matrix
cm = np.array(confusion_matrix(np.array(labels_test), np.array(preds)))
pandas.DataFrame(cm).to_csv("cm1.csv")