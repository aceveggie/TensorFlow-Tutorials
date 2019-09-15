import tensorflow as tf
tf.enable_eager_execution()
import tempfile
import zipfile
import os
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
import time

l = tf.keras.layers

img_rows, img_cols = 28, 28
num_classes = 10
batch_size = 128

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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


full_keras_model = tf.keras.models.load_model('original_network.h5')


pruned_model = tf.keras.models.load_model('final_pruned_model.h5')

y = []
x = []
correct = 0
total = 0
batch_size = 32
timetaken = []
for each in range(x_test.shape[0]):
    x.append(x_test[each,:,:,:])
    y.append(y_test[each,:])
    if(len(x) == batch_size):
        x = np.array(x)
        y = np.array(y)
        # print(x.shape, y.shape)
        start_time = time.time()
        y_pred = full_keras_model.predict(x)
        end_time = time.time()
        y = np.argmax(y,axis=1)
        y_pred = np.argmax(y_pred,axis=1)        
        # print(y, y_pred)
        # print(np.sum(y == y_pred))
        correct += np.sum(y == y_pred)
        total += batch_size
        timetaken.append(end_time-start_time)
        # run inference
        x = []
        y = []

print(correct/float(total),'accuracy')
print('avg time taken for full model....',sum(timetaken)/float(len(timetaken)),'secs')


y = []
x = []
correct = 0
total = 0
batch_size = 32
timetaken = []
for each in range(x_test.shape[0]):
    x.append(x_test[each,:,:,:])
    y.append(y_test[each,:])
    if(len(x) == batch_size):
        x = np.array(x)
        y = np.array(y)
        # print(x.shape, y.shape)
        start_time = time.time()
        y_pred = pruned_model.predict(x)
        end_time = time.time()
        y = np.argmax(y,axis=1)
        y_pred = np.argmax(y_pred,axis=1)        
        # print(y, y_pred)
        # print(np.sum(y == y_pred))
        correct += np.sum(y == y_pred)
        total += batch_size
        timetaken.append(end_time-start_time)
        # run inference
        x = []
        y = []

print(correct/float(total),'accuracy')
print('avg time taken for pruned model....',sum(timetaken)/float(len(timetaken)),'secs')
