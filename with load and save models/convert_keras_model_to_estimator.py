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
keras_model_path = './original_cifar_network.h5'
saved_keras_model = tf.keras.models.load_model(keras_model_path)

tf.keras.estimator.model_to_estimator(
    keras_model=saved_keras_model,
    custom_objects=None,
    model_dir='./estimator_dir/',
    config=None
)

print('done')