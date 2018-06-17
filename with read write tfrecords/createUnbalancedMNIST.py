import tensorflow as tf
import numpy as np
import input_data
import scipy
import scipy.misc

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

superMat = np.eye(10)

def writeMyTFRecords(trX, trY, teX, teY, tfrecords_train_filename, tfrecords_test_filename):
	print ('now writing tfrecords')

	trainWriter = tf.python_io.TFRecordWriter(tfrecords_train_filename)
	testWriter = tf.python_io.TFRecordWriter(tfrecords_test_filename)

	print('train')
	idd = 0
	for eachImg in zip (trX, trY):
		img, label = eachImg[0], eachImg[1]
		img = np.reshape(img, (28, 28))
		width, height, channels = img.shape[0], img.shape[1], 0
		
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		img = np.uint8(img * 255.0)
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		
		label = np.argmax(label, 0)
		
		idd += 1
		if(label == 0):
			label = 0
		elif(label == 7):
			label = 1
			if(idd % 5 == 0):
				pass
			else:
				continue
		else:
			continue

		print(label)
	

		img = img.tostring()
		
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'channels':_int64_feature(channels),
			'image_raw': _bytes_feature(img),
			'label_raw': _int64_feature(label)}))
		trainWriter.write(example.SerializeToString())

	print('test')
	idd = 0
	for eachImg in zip (teX, teY):
		img, label = eachImg[0], eachImg[1]
		img = np.reshape(img, (28, 28))
		width, height, channels = img.shape[0], img.shape[1], 0
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		img = np.uint8(img * 255.0)
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		#print('label', label, np.argmax(label, 0))
		label = np.argmax(label, 0)

		idd += 1
		if(label == 0):
			label = 0
		elif(label == 7):
			label = 1
			if(idd % 5 == 0):
				pass
			else:
				continue
		else:
			continue

		print(label)
		img = img.tostring()

		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'channels':_int64_feature(channels),
			'image_raw': _bytes_feature(img),
			'label_raw': _int64_feature(label)}))
		testWriter.write(example.SerializeToString())

	trainWriter.close()
	testWriter.close()

writeMyTFRecords(trX, trY, teX, teY, "mnistTrain.tfrecords", "mnistTest.tfrecords")
