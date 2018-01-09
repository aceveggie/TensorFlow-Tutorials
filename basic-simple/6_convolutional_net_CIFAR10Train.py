import tensorflow as tf
import numpy as np
import pickle
import random
import cv2

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
        labelList.append(superLabelMat[labelTrainArray[i]])
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
        labelList.append(superLabelMat[labelTestArray[i]])
        # print(labelTestArray[i], superLabelMat[labelTestArray[i]])

    dataTestArray = np.array(imgList)
    labelTestArray = np.array(labelList)
    
    print(dataTrainArray.shape, labelTrainArray.shape, dataTestArray.shape, labelTestArray.shape)
    return dataTrainArray, labelTrainArray, dataTestArray, labelTestArray

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

dataTrainArray, labelTrainArray, dataTestArray, labelTestArray = readCIFAR10(["C:\\Users\\jason\\Desktop\\TensorFlow-Tutorials\\cifar-10-batches-py\\data_batch_1", "C:\\Users\\jason\\Desktop\\TensorFlow-Tutorials\\cifar-10-batches-py\\data_batch_2"])

print(dataTrainArray.mean(), dataTrainArray.std(), dataTestArray.mean(), dataTestArray.std())

# for i in range(5):
#     newIdx = random.randint(1,10000)
#     img = dataTrainArray[newIdx, :, :, :]
#     label = labelTrainArray[newIdx,:]
#     print(label)
#     print(img.shape, img.max(), img.min(), img.std())
#     cv2.imwrite("img.jpg", img)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)

# for i in range(5):
#     newIdx = random.randint(1,10000)
#     img = dataTestArray[newIdx, :, :, :]
#     label = labelTestArray[newIdx,:]
#     print(label)
#     print(img.shape, img.max(), img.min(), img.std())
#     cv2.imwrite("img.jpg", img)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 3, 32]) # int((32 - 3 + 1)/2) = 15 x 15 x 32
w2 = init_weights([5, 5, 32, 64]) # int((15 - 5 + 1)/2) = 5 x 5 x 64
w3 = init_weights([5, 5, 64, 128]) # int((5 - 5 + 1)/2) = 5 x 5 x 64
w4 = init_weights([128 * 4 * 4 , 625])
w_o = init_weights([625, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)
predict_op = tf.argmax(py_x, 1)

tf.summary.scalar('cost value', cost)
saver = tf.train.Saver()

sess = tf.Session()
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(60):

    imgList = []
    labelList = []
    batchID = 0
    for eachIteration in range(dataTrainArray.shape[0]):

        imgList.append(dataTrainArray[eachIteration,:,:,:])
        labelList.append(labelTrainArray[eachIteration,:])

        if(len(imgList) == 16):
            imgList = np.array(imgList)
            imgList = (imgList - 120)/64.0
            labelList = np.array(labelList)
            
            sess.run(train_op, feed_dict={X: imgList, Y: labelList,
                                      p_keep_conv: 0.8, p_keep_hidden: 0.5})

            
            outputY = sess.run(predict_op, feed_dict={X: imgList, Y: labelList,
                                      p_keep_conv: 1, p_keep_hidden: 1.0})
            curCost = sess.run(cost, feed_dict={X: imgList, Y: labelList,
                                      p_keep_conv: 1, p_keep_hidden: 1.0})

            
            if(batchID % 160 == 0):
                print('inputs:', imgList.shape, labelList.shape)
                labelList = np.argmax(labelList, axis= 1)
                print ('epoch: ', i, 'batchID:', batchID, 'curCost: ',curCost, 'accuracy: ', 100.0 * np.mean(outputY == labelList))

            imgList = []
            labelList = []
            batchID += 1
            continue

saver.save(sess, '.\\my-model-cifar10.meta')
