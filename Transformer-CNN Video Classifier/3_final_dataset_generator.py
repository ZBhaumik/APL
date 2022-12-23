import os
import io
import imageio
import ipywidgets
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import pandas as pd
import random

#The purpose of this program is to take your randomized data list, and turn it into a lot of
#different datasets. You can alter the test-train splits here, and get multiple of each split if
#you want.

#Here, you alter the number of divisions and test_splits for the data.
#The number of divisions refers to different test/train split ratios,
#for example number_of_divisions = 2 would give you a 5% and 10% split,
#increasing in %s of 5.
#Number_of_test_splits refers to how many random datasets of each test/train ratio you want
number_of_divisions = 4
number_of_test_splits = 5

#Put the paths to your randomized csv file and np array here.
full_df = pd.read_csv("randomlist.csv")
full_data = np.load("randomdata.npy")

dataset_path = os.listdir('dataset/train')
label_types = os.listdir('dataset/train')

TEST_SPLIT1 = 100
val_split = 110
splitter = 1
while splitter <= number_of_divisions:
    iter = 1
    while iter <= number_of_test_splits:
        tester = 100 - (5*splitter)
        rooms = []
        rooms2 = []
        rooms3 = []
        arr1 = []
        arr2 = []
        arr3 = []
        counter = 0
        for item in dataset_path:
            all_rooms = os.listdir('dataset/train' + '/' + item)
            for room in all_rooms:
                qx = random.randint(1,100)
                if(qx>tester):
                    rooms2.append((full_df["tag"][counter], full_df["video_name"][counter]))
                    arr2.append(full_data[counter])
                else:
                    qy = random.randint(1,100)
                    if(qy>val_split):
                        rooms3.append((full_df["tag"][counter], full_df["video_name"][counter]))
                        arr3.append(full_data[counter])
                    else:
                        rooms.append((full_df["tag"][counter], full_df["video_name"][counter]))
                        arr1.append(full_data[counter])
                counter+=1

        train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
        df = train_df.loc[:,['video_name','tag']]
        df
        df.to_csv('train' + str(5*splitter) + '_' + str(iter) + '.csv')
        test_df = pd.DataFrame(data=rooms2, columns=['tag', 'video_name'])
        df = test_df.loc[:,['video_name','tag']]
        df
        df.to_csv('test' + str(5*splitter) + '_' + str(iter) + '.csv')
        
        arr1 = np.array(arr1)
        np.save('traindata' + str(5*splitter) + '_' + str(iter) + '.npy', arr1)
        arr2 = np.array(arr2)
        np.save('testdata' + str(5*splitter) + '_' + str(iter) + '.npy', arr2)
        
        iter+=1
    splitter+=1