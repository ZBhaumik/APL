import os
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
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

dataset_directory = "dataset/train"
classes_list = os.listdir('dataset/train')
maina = np.load("brush_hair.npy")
mainb = np.load("brush_hair_.npy")
print(maina.shape)
print(mainb.shape)

for class_index, class_name in enumerate(classes_list):
    if(class_index>0):
        maina = np.append(maina, np.load(class_name+".npy"))
        mainb = np.append(mainb, np.load(class_name+"_.npy"))
        print(class_index)

np.save("combineddata.npy", maina)
np.save("combinedlabels.npy", mainb)
print(maina.shape)
print(mainb.shape)