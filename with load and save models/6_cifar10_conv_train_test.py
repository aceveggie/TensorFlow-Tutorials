import tensorflow as tf
import numpy as np
import pickle
import random

# tf.enable_eager_execution()

import tempfile
import zipfile
import os

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


l = tf.keras.layers


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, len(np.unique(y_train)))
num_classes = len(np.unique(y_train))

print(x_train.mean(), x_train.std(), x_test.mean(), x_test.std())
x_train = x_train/255.0
x_test = x_test/255.0

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

img_rows, img_cols, img_channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]

input_shape = (img_rows, img_cols, img_channels)
num_classes = y_train.shape[1]

print(x_train.mean(), x_train.std(), x_test.mean(), x_test.std())

model = tf.keras.Sequential(
    [
    l.Conv2D(
        32, 3, padding='valid', use_bias = True, activation=None, input_shape=input_shape),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    tf.keras.layers.ReLU(),

    l.Conv2D(64, 5, padding='valid', use_bias = True, activation=None),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    tf.keras.layers.ReLU(),

    l.Conv2D(64, 3, padding='valid', use_bias = True, activation=None),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    tf.keras.layers.ReLU(),

    l.Flatten(),
    l.Dense(256, activation='relu'),
    l.Dropout(0.2),
    l.Dense(128, activation='relu'),
    l.Dropout(0.2),
    l.Dense(num_classes, activation='softmax')
])


print(model.summary())

logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)
batch_size=32
epochs=1

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
print('done training the original keras model')

# tf keras way of saving
keras_file = './original_cifar_network.h5'

print('Saving tf keras model to: ', keras_file)
tf.keras.models.save_model(model, keras_file, include_optimizer=True)

saved_keras_model = tf.keras.models.load_model('./original_cifar_network.h5')

print('performing keras model evaluation...')
test_results = saved_keras_model.predict(x_test, batch_size = 128)
test_results = np.argmax(test_results, axis=1)
y_test = np.argmax(y_test, axis=1)
print('accuracy keras model evaluation:', np.sum(test_results == y_test)/len(test_results))


# tf way of saving it as a SavedModel
graph = tf.get_default_graph()
sess = tf.keras.backend.get_session()

builder = tf.saved_model.builder.SavedModelBuilder('./saved_model_cifar_10')
signature = predict_signature_def(inputs={'state': model.input},
                                    outputs={t.name: t for t in model.outputs})
builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                        signature_def_map={
                                            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
builder.save()
print('done saving original in savedmodel..')

saved_cifar10_model = tf.saved_model.load("./saved_model_cifar_10")

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = saved_cifar10_model.predict(x_test[:3])
print('predictions shape:', predictions.shape)