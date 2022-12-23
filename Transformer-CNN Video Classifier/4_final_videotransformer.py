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

#This is where you insert the filepaths to the train and test numpy arrays generated in final_videotrainer
TRAIN_NP_ARR = "traindata20_1.npy"
TEST_NP_ARR = "testdata20_1.npy"

#This is where you insert your csv files corresponding to your training and testing split data.
train_df = pd.read_csv("train20_1.csv")
test_df = pd.read_csv("test20_1.csv")

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

train_data = np.load(TRAIN_NP_ARR)
test_data = np.load(TEST_NP_ARR)

train_labels = train_df["tag"].values
train_labels = label_processor(train_labels[..., None]).numpy()
test_labels = test_df["tag"].values
test_labels = label_processor(test_labels[..., None]).numpy()


#Here is the transformer begins.
#The transformer uses positional embedding (read more at:
# https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11).
# Positional embedding is what gives the transformer its temporal nature, as it is now able to 
# establish relationships between the frames. I do not think you will want to change anything here,
# instead you will want to change things in the TransformerEncoder method (see below).
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: (batch_size, frames, num_features)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

#Here is the TransformerEncoder method. You can directly alter the architecture here, whether it is
#adding layers, changing the dropout rates, etc. You certainly are much more familiar with ML than I am,
#and so I expect you will be able to find great performance by fine-tuning the more advanced parameters.
#(I did not experiment with too much, just dropout rates, the Dense Layer architecture,
# and the LayerNormalization.)
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim, activation=tf.nn.gelu)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

#Compiles the transformer model. This is another place where you can tune more settings. You can
#experiment with different optimizers (the only one that gave me good results was Adam),
# weight decay, different dropouts, pooling layer architecture, learning rate scheduling, etc.
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
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.75)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer = keras.optimizers.Adam(lr=0.00003), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

#This is where the program "begins" per se, where the model begins training. You may want to alter the
#validation_split depending on how the model is doing with regards to the val_acc.
def run_experiment():
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    model = get_compiled_model()
    history = model.fit(
        train_data,
        train_labels,
        shuffle = True,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    return model

trained_model = run_experiment()

#This is the action classification method, you feed it the path to a video,
#and the model will give you a probability distribution with ALL classes to show you
#what % chance it thinks it could be it.
def predict_action(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(path)
    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features)[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

#Takes the images that you fed the neural network and turns them into a GIF.
#It is helpful when running into incorrect classifications, to understand what
#exactly about the behaviour of the video caused the false positive.
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")

#Tests the trained model on n (n=5 here) videos from the training set.
test_counter = 0
while test_counter < 5:
    test_video = np.random.choice(test_df["video_name"].values.tolist())
    print(f"Test video path: {test_video}")
    test_frames = predict_action(test_video)
    to_gif(test_frames[:MAX_SEQ_LENGTH])
    test_counter+=1

#Builds a confusion matrix with the test data, saves it as cm1.csv
preds = trained_model.predict(test_data)
preds = np.argmax(preds, axis=-1)
labels_test = test_labels.flatten()
print(preds)
print(labels_test)
from sklearn.metrics import confusion_matrix
cm = np.array(confusion_matrix(np.array(labels_test), np.array(preds)))
pd.DataFrame(cm).to_csv("cm1.csv")