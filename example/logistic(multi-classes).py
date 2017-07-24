import pandas as pd
import numpy as np
import tensorflow as tf

'''
训练数据是28×28的像素图片，保存在‘字母.npz’里
'''


#转换成one-hot
def Trans(y):
    m = y.shape[0]
    mat = np.zeros((m,10))
    for i in range(m):
        mat[i][y[i]] = 1
    return mat

#load
d = np.load('字母.npz')
X_train,X_test,X_va = d['X_train'],d['X_test'],d['X_va']
y_train,y_test,y_va = d['y_train'],d['y_test'],d['y_va']

#平铺pixel
xx_train = np.reshape(X_train,(X_train.shape[0],28*28))
xx_test = np.reshape(X_test,(X_test.shape[0],28*28))
xx_va = np.reshape(X_va,(X_va.shape[0],28*28))

#转成one-hot
y_train = Trans(y_train)
y_test = Trans(y_test)
y_va = Trans(y_va)

#生成占位符
X = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])

#创建权重和偏差值
W = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))

#假设函数
h = tf.nn.softmax(tf.matmul(X,W)+b)
#用相对熵作为Loss
cross_contropy = -tf.reduce_sum(y*tf.log(h))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_contropy)
init = tf.global_variables_initializer()

#mini-batch gradient descent
with tf.Session() as sess:
    sess.run(init)
    itr = 5
    for i in range(100000):
        x = xx_train[itr-5:itr,:]
        yy = y_train[itr-5:itr,:]
        #yy = np.reshape(yy,[yy.shape[0],10])
        sess.run(train_step,feed_dict={X:x,y:yy})
        itr= itr+5
    correct_predict = tf.equal(tf.argmax(h,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    print('\ntrain Accuracy =',accuracy.eval(session=sess,feed_dict={X:xx_train,y:y_train}))
    print('\ntest Accuracy =',accuracy.eval(session=sess,feed_dict={X:xx_test,y:y_test}))
    print('\nva Accuracy =',accuracy.eval(session=sess,feed_dict={X:xx_va,y:y_va}))
    


