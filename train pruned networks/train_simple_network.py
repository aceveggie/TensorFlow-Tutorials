# code borrowed from https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras

import tensorflow as tf
tf.enable_eager_execution()

import tempfile
import zipfile
import os

# prepare training data
batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# define keras model.

l = tf.keras.layers

# inputs = 28 x 28 x 1, filters = (5 x 5 x 32) + 32 biases = 832 params
# inputs = 28 x 28 x 32, maxpooling = 0 params
# inputs = 14 x 14 x 32, batch normalization = 4 x 32 = 128 params
# inputs = 14 x 14 x 32, filters = (5 x 5 x 64 x 32) + 64 = 51264 params
# inputs = 14 x 14 x 64, maxpooling = 0 params
# inputs = 7 x 7 x 64, flatten = 0 params
# inputs = 3136, fc out = 1024, 10 biases = 3211328 params
# inputs = 1024, fc out = 10, 10 biases  = 10250 params
# output = 10.

model = tf.keras.Sequential([
    l.Conv2D(
        32, 5, padding='same', use_bias = True, activation='relu', input_shape=input_shape),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    l.Conv2D(64, 5, padding='same', use_bias = True, activation='relu'),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    l.Dense(1024, activation='relu'),
    l.Dropout(0.4),
    l.Dense(num_classes, activation='softmax')
])

print(model.summary())

logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)


model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Backend agnostic way to save/restore models
keras_file = './original_network.h5'
print('Saving model to: ', keras_file)
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('done training the original model')