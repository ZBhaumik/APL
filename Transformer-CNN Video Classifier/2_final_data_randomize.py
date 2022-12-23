from cv2 import split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
import tensorflow as tf
import imageio
import cv2


straightarr = np.load("train_data.npy")
straightcsv = pd.read_csv("full_list.csv")

#You will want to change this parameter to the number of videos that you have.
NUMBER_OF_VIDEOS = 13320
randomized = np.zeros((NUMBER_OF_VIDEOS, 20, 1024))

sequential = list(range(0, NUMBER_OF_VIDEOS))
random.shuffle(sequential)

randomcsv = straightcsv.reindex(sequential)
randomcsv['Unnamed: 0'] = randomcsv['Unnamed: 0'].sort_values(ascending=True).values
randomcsv.to_csv('randomlist.csv', index = False)

i=0
while i < NUMBER_OF_VIDEOS:
    randomized[i] = straightarr[sequential[i]]
    i+=1
np.save('randomdata.npy', randomized)