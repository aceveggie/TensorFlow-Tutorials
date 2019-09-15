import tensorflow as tf
tf.enable_eager_execution()
import tempfile
import zipfile
import os
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
import time

full_keras_model = tf.keras.models.load_model('original_network.h5')
print('loaded the full keras model')

for i, w in enumerate(full_keras_model.get_weights()):
    print(
        "{} -- Total:{}, Zeros: {:.2f}%".format(
            full_keras_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
        )
    )

pruned_model = tf.keras.models.load_model('final_pruned_model.h5')

for i, w in enumerate(pruned_model.get_weights()):
    print(
        "{} -- Total:{}, Zeros: {:.2f}%".format(
            pruned_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
        )
    )