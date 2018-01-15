import cv2
import numpy as np
import pickle
import random
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

superLabelMat = np.eye(10)

def readCIFAR10(fileList):
    print('reading train cifar10')
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
    
    print('reading test cifar')
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

#https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def readWriteSpriteImage(logDir):
	dataTrainArray, labelTrainArray, dataTestArray, labelTestArray = readCIFAR10(["..\\data\\cifar-10-batches-py\\data_batch_1", "..\\data\\cifar-10-batches-py\\data_batch_2"])
	imgList = []
	for eachImg in dataTrainArray:
		imgList.append(eachImg)
	print('individual image is of shape', imgList[0].shape)
	imgList = np.array(imgList)
	spriteImg = images_to_sprite(imgList)
	print('creating big sprite with shape',spriteImg.shape)
	cv2.imwrite(logDir+"sprite.png", spriteImg)
	fp = open(logDir+"metadata.tsv","w")
	idd = 0
	for each in labelTrainArray:
		fp.write(str(idd)+"\t"+str(each)+'\n')
		idd += 1
	fp.close()