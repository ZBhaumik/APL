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
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

full = np.load("b__test_data.npy")
#full2 = np.load("all_data.npy")
#tester = np.load("test_labels.npy")
print(full)
print(full.shape)
print(full[1])
#print(full2.shape)
#print(tester[0])
#print(tester)