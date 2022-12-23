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
np.set_printoptions(suppress=True)

# Setting seed for reproducibility
#SEED = 42
#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
#keras.utils.set_random_seed(SEED)

# DATA
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 51

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

MAX_SEQ_LENGTH = 20
IMG_SIZE = 32
INPUT_SHAPE = (MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 1)
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

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
            #frame = crop_center(frame)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
            frame = frame[:, :, [2, 1, 0]]
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = np.dot(frame[...,:3], [0.1140, 0.5870, 0.2989])
            frame = np.round(frame,2)
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

full_df = pd.read_csv("full_list.csv")

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(full_df["tag"]), mask_token=None
)


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    counter = 0
    hyperframes = np.zeros((6765, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE))

    for idx, path in enumerate(video_paths):
        print(counter)
        frames = load_video(path, MAX_SEQ_LENGTH)
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE))
            frames = np.concatenate(frames, padding)
        hyperframes[counter] = frames
        counter+=1
    return hyperframes, labels


@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    #Preprocess the frames tensors and parse the labels.
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    #Utility function to prepare the dataloader.
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    #if loader_type == "train":
    #    dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

train_videos = np.load("traindata10_1.npy")
test_videos = np.load("testdata10_1.npy")
valid_videos = np.load("valdata10_1.npy")

train_df = pd.read_csv("train10_1.csv")
test_df = pd.read_csv("test10_1.csv")
val_df = pd.read_csv("val10_1.csv")


train_labels = train_df["tag"].values
train_labels = label_processor(train_labels[..., None]).numpy()
test_labels = test_df["tag"].values
test_labels = label_processor(test_labels[..., None]).numpy()
valid_labels = val_df["tag"].values
valid_labels = label_processor(valid_labels[..., None]).numpy()

print(train_videos)
print(train_labels)
trainloader = prepare_dataloader(train_videos, train_labels, "train")
validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
testloader = prepare_dataloader(test_videos, test_labels, "test")

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.relu),
                layers.Dense(units=embed_dim, activation=tf.nn.relu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Train the model.
    _ = model.fit(trainloader, epochs=EPOCHS, validation_data = validloader)

    _, accuracy, top_5_accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model

model = run_experiment()
"""
all_data, all_labels = prepare_all_videos(full_df, "train")
print("bbbbb")
np.save("all_data.npy", all_data)
np.save("all_labels.npy", all_labels)
"""