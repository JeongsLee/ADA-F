import os

import numpy as np
np.random.seed(1234)

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(1234)
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
#import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224

batch_size = 64

dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))



img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

checkpoint_filepath = "./transfer/checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

model = build_model(num_classes=NUM_CLASSES)
model.summary()
epochs = 40  # @param {type: "slider", min:8, max:80}
history = model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=[checkpoint_callback])

loss = np.array(history.history['loss'])
accuracy = np.array(history.history['accuracy'])
val_loss = np.array(history.history['val_loss'])
val_accuracy = np.array(history.history['val_accuracy'])

np.savetxt('./loss_T.txt', loss, delimiter=',')
np.savetxt('./accuracy_T.txt', accuracy, delimiter=',')
np.savetxt('./val_loss_T.txt', val_loss, delimiter=',')
np.savetxt('./val_accuracy_T.txt', val_accuracy, delimiter=',')  

model.save_weights('./checkpoints/transfer/ckpoint_adaf')