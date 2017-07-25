import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X_train,y_train = mnist.train.images[:50000,:],mnist.train.labels[:50000,:]
X_test,y_test = mnist.test.images[:800],mnist.test.labels[:800]

X_tr = tf.placeholder(tf.float32,[None,28*28])
X_te = tf.placeholder(tf.float32,[28*28])

#L1范数
idx = tf.arg_min(tf.reduce_sum(tf.abs(tf.add(X_tr,-X_te)),axis=1),0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    cnt = 0
    for i in range(X_test.shape[0]):
        #第i个图片
        record = X_test[i,:]
        index = sess.run(idx,feed_dict={X_tr:X_train,X_te:record})
        if np.argmax(y_train[index])==np.argmax(y_test[i]):
            cnt = cnt + 1
        if ( i%50==0 and i!=0 ):
            print('[Now test samples :',i,' Accuracy :',(cnt/i))
    print('cnt',cnt) 
    print('Total Accuracy :',(cnt/X_test.shape[0])) 
