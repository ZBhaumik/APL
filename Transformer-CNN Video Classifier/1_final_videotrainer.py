from cgi import test
from tensorflow_docs.vis import embed
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

#Here, you can alter parameters. If you have a higher RAM workstation,
#you'll likely want to increase MAX_SEQ_LENGTH to get higher accuracies.
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 224
EPOCHS = 100

#This is where you insert your csv files corresponding to your training and testing split data.
#With regards to this, program, you will want to have a full csv file of the paths and labels
#to all of your pieces of data, formatted like full_list.csv (attached to the email.)
train_df = pd.read_csv("full_list.csv")

#This method will take the center crop of an image (for this application, it takes the middle 224*224 pixels
#of each frame).
center_crop_layer = layers.CenterCrop(IMG_SIZE, IMG_SIZE)
def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped

#This method is for loading videos and splitting them into frames. The max_frames argument corresponds
#to the MAX_SEQ_LENGTH further down, and allows you to standardize the amount of frames/video.
def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center(frame)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

#This is the feature extractor, which is the heart of how this program works.
#Essentially, it uses the DenseNet121 ImageNet CNN on each image, to convert the 224*224 rgb array into
#an array of 1024 values (features). You can increase the number of pixels (image size), or features
#however I believe this will require a different feature extractor.
def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
)

#This is the main method of the entire program, where it runs through all of your videos
#in your dataset, feeds the frames to the feature extractor, and then saves it as a Numpy array.
def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    counter = -1
    for idx, path in enumerate(video_paths):
        counter+=1
        print(counter)
        # Gather all of the video's frames and add a batch dimension.
        frames = load_video(path, MAX_SEQ_LENGTH)
        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate((frames, padding))
        frames = frames[None, ...]
        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )
                else:
                    temp_frame_features[i, j, :] = 0.0
        frame_features[idx,] = temp_frame_features.squeeze()
    return frame_features, labels

#Same functionality as above method, but this is designed for a single video. Used to test the model
#against the testing split.
def prepare_single_video(frames):
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate((frames, padding))
    frames = frames[None, ...]
    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0
    return frame_features

train_data = prepare_all_videos(train_df, "train")
train_labels = train_df["tag"].values
train_labels = label_processor(train_labels[..., None]).numpy()

#The training and testing data and label numpy arrays are saved using the names below.
np.save("full_data.npy", train_data)
np.save("full_labels.npy", train_labels)