import tensorflow as tf
import numpy as np
import input_data


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
    pyx = tf.sigmoid(pyx)
    return pyx

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_o = init_weights([625, 2])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=py_x)
cost = tf.reduce_mean(cross_entropy)

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

tf.summary.scalar('cost value', cost)

sess = tf.Session()
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

saver.restore(sess, './UnbalancedMNIST.meta')

batchSize = 8

superMat = np.eye(2)

tfrecords_train_filename = "mnistTrain.tfrecords"

imgList = []
labelList = []

costList = []



trainRecordIterator = tf.python_io.tf_record_iterator(path=tfrecords_train_filename)
idd = 0

total = 0
totalPos = 0
totalNeg = 0
tp = 0
correct = 0
fp = 0
tn = 0
fn = 0

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
        img = img.reshape(width, height, 1)
    
    idd += 1
    curLabel = superMat[label]
    labelList.append(curLabel)
    imgList.append(img)

    if(len(imgList) == batchSize):
        
        labelList = np.reshape(labelList, (batchSize, 2))

        imgList = np.array(imgList)

        # call the train method
        # print(labelList, labelList.shape)
        # print(imgList.shape, imgList.mean(), imgList.std(), imgList.max(), imgList.min())
        sess.run(train_op, feed_dict={X: imgList, Y: labelList, p_keep_conv: 0.5, p_keep_hidden: 0.5})
        curCost = sess.run(cost, feed_dict={X: imgList, Y: labelList, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        predictedList = sess.run(py_x, feed_dict={X: imgList, Y: labelList, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        costList.append(curCost)
        print(predictedList.shape)
        for i in zip(predictedList, labelList):
            print(np.argmax(i[0], 0), np.argmax(i[1], 0))
            predictedLabel = np.argmax(i[0], 0)
            originalLabel = np.argmax(i[1], 0)
            #print("predicted: ", predictedLabel, 'original', originalLabel)
            #print('----')
            total += 1
            if(originalLabel == 1):
                totalPos += 1

            if(originalLabel == 0):
                totalNeg += 1

            if(predictedLabel == 1 and originalLabel == 1):
                tp += 1
                correct += 1
            elif(predictedLabel == 1 and originalLabel == 0):
                fp += 1
            elif(predictedLabel == 0 and originalLabel == 0):
                tn += 1
                correct += 1
            elif(predictedLabel == 0 and originalLabel == 1):
                fn += 1

        labelList = []
        imgList = []
    

accuracy = 100.0 * (float(correct)/total)
precision = float(tp)/(tp + fp)
recall = float(tp)/(tp + fn)
f1score = 2.0 * float(precision * recall)/(precision + recall)

print('accuracy = ', accuracy)
print('---')
print('precision = ', precision)
print('recall = ', recall)
print('---')
print('f1score = ',f1score)
print('---')
print('tp = ', tp)
print('fp = ', fp)
print('tn = ', tn)
print('fn = ', fn)
print('---')
print('totalPos = ', totalPos)
print('totalNeg = ', totalNeg)
print('---')
print('total = ', total)