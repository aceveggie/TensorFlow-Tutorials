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



def variable_summaries(var):
    ''' get your variable summaries for visualization '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# def init_weights(shape, nameScope):
#     with tf.name_scope(nameScope) as scope:
#         w = tf.Variable(tf.random_normal(shape, stddev=0.01))
#     return w

def init_weights(shape, name, type='conv'):
    initializer=tf.contrib.layers.xavier_initializer()
    if(type =='conv'):
        w = tf.get_variable(name=name, shape=shape, initializer= initializer)
    elif(type == 'b'):
        w = tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1))
    elif(type == 'fc'):
        w = tf.get_variable(name=name, shape=shape, initializer= initializer)
    return w

def model(X, weights, biases, p_keep_conv, p_keep_hidden):

    conv1, conv2, conv3, w4, w_o = weights
    b1, b2, b3, b4, b_o = biases
   
    l1a = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(X, conv1, [1, 1, 1, 1], 'SAME') + b1))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(l1, conv2, [1, 1, 1, 1], 'SAME') + b2))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(l2, conv3, [1, 1, 1, 1], 'SAME') + b3))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    #l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.contrib.layers.flatten(l3)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(l3, w4) + b4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o) + b_o
    return pyx

dataTrainArray, labelTrainArray, dataTestArray, labelTestArray = readCIFAR10(["..\\data\\cifar-10-batches-py\\data_batch_1", "..\\data\\cifar-10-batches-py\\data_batch_2"])

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

with tf.name_scope('inputs') as scope:
    X = tf.placeholder("float", [None, 32, 32, 3], name='X')
    Y = tf.placeholder("float", [None, 10], name='Y')

with tf.name_scope('inputReshaped') as scope:
    XReshaped = tf.reshape(X, [-1, 32, 32, 3])
    tf.summary.image('input', XReshaped, 5)

# initialize all the tensors

with tf.name_scope('conv1'):
    conv1 = init_weights([3, 3, 3, 32], 'conv1', type='conv') # 16 x 16 x 32 o/p
    variable_summaries(conv1)

with tf.name_scope('b1'):
    b1 = init_weights([1, 1, 1, conv1.get_shape().as_list()[3]], 'b1', type='b')
    variable_summaries(b1)

with tf.name_scope('conv2'):
    conv2 = init_weights([5, 5, 32, 64], 'conv2', type='conv') # 8 x 8 x 64 o/p
    variable_summaries(conv2)

with tf.name_scope('b2'):
    b2 = init_weights([1, 1, 1, conv2.get_shape().as_list()[3]], 'b2', type='b')
    variable_summaries(b2)

with tf.name_scope('conv3'):
    conv3 = init_weights([5, 5, 64, 128], 'conv3', type='conv') # 4 x 4 x 128 o/p # flatten this
    variable_summaries(conv3)

with tf.name_scope('b3'):
    b3 = init_weights([1, 1, 1, conv3.get_shape().as_list()[3]], 'b3', type='b')
    variable_summaries(b3)

with tf.name_scope('w4'):
    w4 = init_weights([128 * 4 * 4 , 625], name='w4', type='fc') # 625 o/p
    variable_summaries(w4)

with tf.name_scope('b4'):
    b4 = init_weights([1, w4.get_shape().as_list()[1]], name='b4', type='b')
    variable_summaries(b4)

with tf.name_scope('w_o'):
    w_o = init_weights([625, 10], name='w_o', type='fc') # 10 o/p
    variable_summaries(w_o)

with tf.name_scope('b_o'):
    b_o = init_weights([1, w_o.get_shape().as_list()[1]], name='b_o', type='b')
    variable_summaries(b_o)

weights = [conv1, conv2, conv3, w4, w_o]
biases = [b1, b2, b3, b4, b_o]

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, weights, biases, p_keep_conv, p_keep_hidden)

with tf.name_scope('cross_entropy'):
    calculatedError = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
    with tf.name_scope("total"):
        cost = tf.reduce_mean(calculatedError)
tf.summary.scalar("cross_entropy", cost)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # avg accuracy
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
saver = tf.train.Saver()
sess = tf.Session()

# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
# test_writer = tf.summary.FileWriter('./logs/test')
# tf.global_variables_initializer().run(session=sess)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

saver.restore(sess,"my-model-cifar10-batchNormalized"+str(27))


imgList = []
labelList = []
batchID = 0
avgAccuracyList = []
for eachIteration in range(dataTestArray.shape[0]):

    imgList.append(dataTestArray[eachIteration,:,:,:])
    labelList.append(labelTestArray[eachIteration,:])

    if(len(imgList) == 16):
        imgList = np.array(imgList)
        # feature scaling
        imgList = (imgList - imgList.mean())/imgList.std() # obtained from training dataset or cur image
        labelList = np.array(labelList)
        imgList = np.float64(imgList)
        # print(imgList.dtype)
        
        transformedY = np.argmax(labelList, 1)
        # get the prediction

        outputY = sess.run(predict_op, feed_dict={X: imgList, Y: labelList,
                                  p_keep_conv: 1, p_keep_hidden: 1.0})

        labelList = np.argmax(labelList, axis= 1)
        curAccuracy = 100.0 * np.mean(outputY == labelList)
        print('batchID:', batchID, 'accuracy: ', curAccuracy)
        avgAccuracyList.append(curAccuracy)

        imgList = []
        labelList = []
        batchID += 1

print('avg accuracy on test set: ', sum(avgAccuracyList)/float(len(avgAccuracyList)))