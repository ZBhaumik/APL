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

"""
dataset_path = os.listdir('dataset/train')
label_types = os.listdir('dataset/train')
rooms = []
for item in dataset_path:
    all_rooms = os.listdir('dataset/train' + '/' + item)
    for room in all_rooms:
        rooms.append((item, str('dataset/train' + '/' + item) + '/' + room))

full_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
df = full_df.loc[:,['video_name','tag']]
df
df.to_csv('full_list.csv')
"""

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128
EPOCHS = 20

full_df = pd.read_csv("randomcsv.csv")
print(full_df)
print(f"Total videos: {len(full_df)}")

center_crop_layer = tf.keras.layers.CenterCrop(IMG_SIZE, IMG_SIZE)
def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped

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

# Label preprocessing with StringLookup.
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(full_df["tag"]), mask_token=None
)
#print(label_processor.get_vocabulary())

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )
    counter = 1

    for idx, path in enumerate(video_paths):
        print(counter)
        counter+=1
        print(path)
        frames = load_video(path)
        
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate(frames, padding)
        frames = frames[None, ...]

        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

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

full_data = np.load("randomnpy.npy")#prepare_all_videos(full_df, "train")#
print(full_data.shape)

"""
dataset_path = os.listdir('dataset/train')
label_types = os.listdir('dataset/train')
TEST_SPLIT1 = 100
splitter = 1
while splitter <= 4:#4:
    iter = 1
    while iter <= 5:#5:
        tester = 100 - (5*splitter)
        rooms = []
        rooms2 = []
        arr1 = []
        arr2 = []
        counter = 0
        for item in dataset_path:
            all_rooms = os.listdir('dataset/train' + '/' + item)
            for room in all_rooms:
                qx = random.randint(1,100)
                if(qx>tester):
                    rooms2.append((full_df["tag"][counter], full_df["video_name"][counter]))
                    #rooms2.append((item, str('dataset/train' + '/' + item) + '/' + room))
                    arr2.append(full_data[counter])
                else:
                    rooms.append((full_df["tag"][counter], full_df["video_name"][counter]))
                    #rooms.append((item, str('dataset/train' + '/' + item) + '/' + room))
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
"""

#train_data, train_labels = np.load("train_data.npy"), np.load("train_labels.npy")#prepare_all_videos(train_df, "train")#
#np.save("train_data.npy", train_data)
#np.save("train_labels.npy", train_labels)

train_df = pd.read_csv("train20_2.csv")
test_df = pd.read_csv("test20_2.csv")

train_data= np.load("traindata20_2.npy")
train_labels = train_df["tag"].values
train_labels = label_processor(train_labels[..., None]).numpy()

test_data = np.load("testdata20_2.npy")
test_labels = test_df["tag"].values
test_labels = label_processor(test_labels[..., None]).numpy()

full_labels = full_df["tag"].values
full_labels = label_processor(full_labels[..., None]).numpy()

print(full_labels)
print(test_labels)
print(train_labels)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation=tf.nn.gelu), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = len(label_processor.get_vocabulary())

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    #CHANGED FROM SOFTMAX TO SIGMOID
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def run_experiment():
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.1,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model

trained_model = run_experiment()

def prepare_single_video(frames):
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate(frames, padding)

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

def predict_action(path):
    class_vocab = label_processor.get_vocabulary()
    frames = load_video(path)
    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features)[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")

iii = 0

while (iii < 5):
    test_video = np.random.choice(test_df["video_name"].values.tolist())
    print(f"Test video path: {test_video}")
    test_frames = predict_action(test_video)
    iii+=1