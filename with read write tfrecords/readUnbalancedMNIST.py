import tensorflow as tf
import numpy as np
import scipy
import scipy.misc

import time
mDictTrain = {}
mDictTest = {}

def readMyTFRecords(tfrecords_train_filename, tfrecords_test_filename):

	print ('now reading tfrecords')
	time.sleep(3)

	trainRecordIterator = tf.python_io.tf_record_iterator(path=tfrecords_train_filename)
	testRecordIterator = tf.python_io.tf_record_iterator(path=tfrecords_test_filename)
	idd = 0
	print('train')
	for eachRecord in trainRecordIterator:
		example = tf.train.Example()
		example.ParseFromString(eachRecord)

		height = int(example.features.feature['height']
			.int64_list
			.value[0])

		width = int(example.features.feature['width']
			.int64_list
			.value[0])

		channels = int(example.features.feature['channels']
			.int64_list
			.value[0])

		img = (example.features.feature['image_raw']
			.bytes_list
			.value[0])

		label = int(example.features.feature['label_raw']
			.int64_list
			.value[0])

		img = np.fromstring(img, dtype=np.uint8)
		if(channels>0):
			img = img.reshape(width, height, channels)
		else:
			img = img.reshape(width, height)
		# print(label)
		# print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)

		# print('label', label)
		curLabel = label
		try:
			mDictTrain[curLabel] += 1
		except:
			mDictTrain[curLabel] = 1

	for eachRecord in testRecordIterator:
		example = tf.train.Example()
		example.ParseFromString(eachRecord)

		height = int(example.features.feature['height']
			.int64_list
			.value[0])

		width = int(example.features.feature['width']
			.int64_list
			.value[0])

		channels = int(example.features.feature['channels']
			.int64_list
			.value[0])

		img = (example.features.feature['image_raw']
			.bytes_list
			.value[0])

		label = int(example.features.feature['label_raw']
			.int64_list
			.value[0])

		img = np.fromstring(img, dtype=np.uint8)
		if(channels>0):
			img = img.reshape(width, height, channels)
		else:
			img = img.reshape(width, height)
		#print(label)
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)

		#print('label', label)
		curLabel = label
		try:
			mDictTest[curLabel] += 1
		except:
			mDictTest[curLabel] = 1
readMyTFRecords("mnistTrain.tfrecords", "mnistTest.tfrecords")


print('---')
for k in list(mDictTrain.keys()):
	print(k, mDictTrain[k])

for k in list(mDictTest.keys()):
	print(k, mDictTest[k])

print('---')