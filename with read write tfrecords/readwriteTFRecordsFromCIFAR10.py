import tensorflow as tf
import numpy as np
import scipy
import pickle
import scipy.misc
import os
import time
import matplotlib.pyplot as plt


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

superLabelMat = np.eye(10)

def readCIFAR10(fileList):
    print('reading train')
    dataDict = unpickle(fileList[0])
    # for each in dataDict.keys():
    #     print(str(each.decode("utf-8")))

    dataTrainArray = dataDict[b'data']
    labelTrainArray = dataDict[b'labels']

    imgList = []
    labelList = []
    for i in range(dataTrainArray.shape[0]):
        newImg = np.zeros((32, 32, 3))
        imgR = np.reshape(dataTrainArray[i, 0:1024], (32, 32))
        imgG = np.reshape(dataTrainArray[i, 1024:(1024*2)], (32, 32))
        imgB = np.reshape(dataTrainArray[i, (1024*2):(1024*3)], (32, 32))
        # print(imgR.shape, imgG.shape, imgB.shape)
        newImg[:,:, 0], newImg[:,:, 1], newImg[:,:, 2] = imgR, imgG, imgB
        imgList.append(newImg)
        labelList.append(labelTrainArray[i])
        # print(labelTrainArray[i], superLabelMat[labelTrainArray[i]])

    dataTrainArray = np.array(imgList)
    labelTrainArray = np.array(labelList)
    
    print('reading test')
    dataDict = unpickle(fileList[1])
    dataTestArray = dataDict[b'data']
    labelTestArray = dataDict[b'labels']

    imgList = []
    labelList = []
    for i in range(dataTestArray.shape[0]):
        newImg = np.zeros((32, 32, 3))
        imgR = np.reshape(dataTestArray[i, 0:1024], (32, 32))
        imgG = np.reshape(dataTestArray[i, 1024:(1024*2)], (32, 32))
        imgB = np.reshape(dataTestArray[i, (1024*2):(1024*3)], (32, 32))
        # print(imgR.shape, imgG.shape, imgB.shape)
        newImg[:,:, 0], newImg[:,:, 1], newImg[:,:, 2] = imgR, imgG, imgB
        imgList.append(newImg)
        labelList.append(labelTestArray[i])
        # print(labelTestArray[i], superLabelMat[labelTestArray[i]])

    dataTestArray = np.array(imgList)
    labelTestArray = np.array(labelList)
    
    print(dataTrainArray.shape, labelTrainArray.shape, dataTestArray.shape, labelTestArray.shape)
    return dataTrainArray, labelTrainArray, dataTestArray, labelTestArray

############################ TF RECORD WRITER ############################
#def writeMyTFRecords(dataTrainArray, labelTrainArray, dataTestArray, labelTestArray, tfrecords_train_filename, tfrecords_test_filename)
def writeMyTFRecords(trX, trY, teX, teY, tfrecords_train_filename, tfrecords_test_filename):
	# mnist = input_data.read_data_sets("./", one_hot=False)
	# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	# trX = trX.reshape(-1, 50, 50, 1)
	# teX = teX.reshape(-1, 50, 50, 1)
	print ('now writing tfrecords')
	time.sleep(3)

	trainWriter = tf.python_io.TFRecordWriter(tfrecords_train_filename)
	testWriter = tf.python_io.TFRecordWriter(tfrecords_test_filename)

	idd = 0
	print('train')
	for eachImg in zip (trX, trY):
		img, label = eachImg[0], eachImg[1]
		
		width, height, channels = img.shape[0], img.shape[1], img.shape[2]
		img = np.uint8(img)
		print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		print('label', label)
		#exit()
		img = img.tostring()
		label = label.tostring()
		
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'channels':_int64_feature(channels),
			'image_raw': _bytes_feature(img),
			'label_raw': _bytes_feature(label)}))
		trainWriter.write(example.SerializeToString())
		idd += 1
		if(idd >= 5):
			break

	idd = 0
	print('test')
	for eachImg in zip (teX, teY):
		img, label = eachImg[0], eachImg[1]
		
		width, height, channels = img.shape[0], img.shape[1], img.shape[2]
		img = np.uint8(img)
		print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		print('label', label)
		#exit()
		img = img.tostring()
		label = label.tostring()

		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'channels':_int64_feature(channels),
			'image_raw': _bytes_feature(img),
			'label_raw': _bytes_feature(label)}))
		testWriter.write(example.SerializeToString())
		idd += 1
		if(idd >= 5):
			break


	trainWriter.close()
	testWriter.close()
	print ('finished writing tf records')

############################ TF RECORD READER ############################
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

		label = (example.features.feature['label_raw']
			.bytes_list
			.value[0])

		img = np.fromstring(img, dtype=np.uint8)
		img = img.reshape(width, height, channels)
		label = np.fromstring(label, dtype=np.uint8)[0]
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		#print('label', label)
		#plt.imshow(img)
		#plt.show()
		idd += 1
		if(idd >= 5):
			break

	idd = 0
	print('test')
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

		label = (example.features.feature['label_raw']
			.bytes_list
			.value[0])

		img = np.fromstring(img, dtype=np.uint8)
		img = img.reshape(width, height, channels)
		label = np.fromstring(label, dtype=np.uint8)[0]
		#print(img.min(), img.max(), img.mean(), img.std(), width, height, channels, img.dtype)
		#print('label', label)
		#plt.imshow(img)
		#plt.show()
		idd += 1
		if(idd >= 5):
			break

	print ('finished reading tf records')

dataTrainArray, labelTrainArray, dataTestArray, labelTestArray = readCIFAR10(["..\\data\\cifar-10-batches-py\\data_batch_1", "..\\data\\cifar-10-batches-py\\data_batch_2"])
print('(train mean, test std, test mean, test std) =',dataTrainArray.mean(), dataTrainArray.std(), dataTestArray.mean(), dataTestArray.std())

tfrecords_train_filename = '..\\data\\cifar10-tfrecords\\CIFAR10_Train.tfrecords'
tfrecords_test_filename = '..\\data\\cifar10-tfrecords\\CIFAR10_Test.tfrecords'

writeMyTFRecords(dataTrainArray, labelTrainArray, dataTestArray, labelTestArray, tfrecords_train_filename, tfrecords_test_filename)
readMyTFRecords(tfrecords_train_filename, tfrecords_test_filename)

pass