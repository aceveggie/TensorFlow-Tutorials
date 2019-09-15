import tensorflow as tf
tf.enable_eager_execution()
import tempfile
import zipfile
import os
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
l = tf.keras.layers

img_rows, img_cols = 28, 28
num_classes = 10
batch_size = 128

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

epochs = 12
num_train_samples = x_train.shape[0]

# calculate the last step (iteration). Prune after you reach this step.
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print('End step: ' + str(end_step))

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=end_step,
                                                   frequency=100)
}

pruned_model = tf.keras.Sequential([
    sparsity.prune_low_magnitude(
        l.Conv2D(32, 5, padding='same', activation='relu'),
        input_shape=input_shape,
        **pruning_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    sparsity.prune_low_magnitude(
        l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    sparsity.prune_low_magnitude(l.Dense(1024, activation='relu'),
                                 **pruning_params),
    l.Dropout(0.4),
    sparsity.prune_low_magnitude(l.Dense(num_classes, activation='softmax'),
                                 **pruning_params)
])

print(pruned_model.summary())

pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    sparsity.UpdatePruningStep()
    ]

print('starting to train a model with pruning parameters....')
pruned_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=0,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

score = pruned_model.evaluate(x_test, y_test, verbose=0)
print('performance on directly trained model (pruned layer by layer)')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('done training a pruned model.......')

checkpoint_file = 'pruned_network.h5'
print('Saving pruned model to: ', checkpoint_file)

################################################################################################################

# saved_model() sets include_optimizer to True by default. Spelling it out here
# to highlight.
# tf.keras.models.save_model(pruned_model, checkpoint_file, include_optimizer=True)

# with sparsity.prune_scope():
#   restored_model = tf.keras.models.load_model(checkpoint_file)

# print('re-training a restored pruned model......')

# restored_model.fit(x_train, y_train,
#                    batch_size=batch_size,
#                    epochs=2,
#                    verbose=0,
#                    callbacks=callbacks,
#                    validation_data=(x_test, y_test))

# score = restored_model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

print('strip the pruned params')

pruned_keras_file = 'final_pruned_model.h5'
final_model = sparsity.strip_pruning(pruned_model)
final_model.summary()
print('saved pruned model to disk....')
tf.keras.models.save_model(final_model, pruned_keras_file, include_optimizer=False)
keras_file = './original_network.h5'

print('model comparison....')

_, zip1 = tempfile.mkstemp('.zip') 
with zipfile.ZipFile(zip1, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(keras_file)
print("Size of the unpruned model before compression: %.2f Mb" % 
      (os.path.getsize(keras_file) / float(2**20)))
print("Size of the unpruned model after compression: %.2f Mb" % 
      (os.path.getsize(zip1) / float(2**20)))

_, zip2 = tempfile.mkstemp('.zip') 
with zipfile.ZipFile(zip2, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(pruned_keras_file)
print("Size of the pruned model before compression: %.2f Mb" % 
      (os.path.getsize(pruned_keras_file) / float(2**20)))
print("Size of the pruned model after compression: %.2f Mb" % 
      (os.path.getsize(zip2) / float(2**20)))