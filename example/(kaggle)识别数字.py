import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from random import sample

#数据预处理
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 28, 28, 1)).astype(np.float32)
    labels = (np.arange(10) == labels[:,None]).astype(np.float32)
    return (dataset, labels)

data = input_data.read_data_sets('MNIST_data/',one_hot=True)
Xte = data.train.images.reshape((-1,28,28,1))
yte = data.train.labels

dataset = pd.read_csv('train.csv',header=None)
dataset = dataset[1:]
X = np.array(dataset[list(range(1,785))],dtype='float32')
y = np.array(dataset[0],dtype='float')

X_train,y_train = reformat(X,y)

'''
X_tr,X_test,y_tr,y_test = train_test_split(X,y,random_state=33,test_size=0.2)
X_train,X_va,y_train,y_va = train_test_split(X_tr,y_tr,random_state=29,test_size=0.25)
X_train,y_train = reformat(X_train,y_train)
X_test,y_test = reformat(X_test,y_test)
X_va,y_va = reformat(X_va,y_va)
'''

#CNN 32 pool 64 pool NN: 2048 1024    

#参数初始化

W1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([49*64,2048],stddev=0.1))
W4= tf.Variable(tf.truncated_normal([2048,1024],stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))

b1 = tf.Variable(tf.zeros([32])+0.5)
b2 = tf.Variable(tf.zeros([64])+0.5)
b3 = tf.Variable(tf.zeros([2048])+0.5)
b4 = tf.Variable(tf.zeros([1024])+0.5)
b5 = tf.Variable(tf.zeros([10])+0.5)


#======================函数定义==========================
def bnorm(h,offset,itr,is_test,is_conv=False):
    ema = tf.train.ExponentialMovingAverage(0.99,itr)
    epsilon = 1e-5
    if is_conv :
        mean,var = tf.nn.moments(h,[0,1,2])
    else:
        mean,var = tf.nn.moments(h,[0])
    update_moving_average = ema.apply([mean,var])
    '''
    Note : 
    1、如果是测试的，那么均值和方差就使用之前算的平均(ema.average(...))
    2、如果不是测试，则使用当前mini-batch的均值和方差
    3、不必担心因为用了normalization而破坏了刚学到的特征，γ和β(offset)会解决这个问题
    4、γ和β是model自动学习出来的
    '''
    m = tf.cond(is_test,lambda: ema.average(mean),lambda: mean)
    v = tf.cond(is_test,lambda: ema.average(var),lambda: var)
    #执行bn操作
    h_bn = tf.nn.batch_normalization(h,m,v,offset,None,epsilon)
    return (h_bn,update_moving_average)

def accuracy(X,y):
    return (100*np.sum(np.argmax(X,1)==np.argmax(y,1))/y.shape[0])

def Mode(X):
    #lay1
    h_1 = tf.nn.conv2d(X,W1,[1,1,1,1],padding='SAME')
    h1,update1 = bnorm(h_1,b1,itr,is_test,True)
    lay1 = tf.nn.relu(h1)
    pool1 = tf.nn.max_pool(lay1,[1,2,2,1],[1,2,2,1],padding='SAME')

    #lay2
    h_2 = tf.nn.conv2d(pool1,W2,[1,1,1,1],padding='SAME')
    h2,update2 = bnorm(h_2,b2,itr,is_test,True)
    lay2 = tf.nn.relu(h2)
    pool2 = tf.nn.max_pool(lay2,[1,2,2,1],[1,2,2,1],padding='SAME')

    #lay3
    Shape = [-1,W3.get_shape().as_list()[0]]
    h_3 = tf.matmul(tf.reshape(pool2,shape=Shape),W3)
    h3,update3 = bnorm(h_3,b3,itr,is_test)
    lay_3 = tf.nn.relu(h3)
    lay3 = tf.nn.dropout(lay_3,keep_drop)

    #lay4
    h_4 = tf.matmul(lay3,W4)
    h4,update4 = bnorm(h_4,b4,itr,is_test)
    lay_4 = tf.nn.relu(h4)
    lay4 = tf.nn.dropout(lay_4,keep_drop)

    #output
    h5 = tf.matmul(lay4,W5) + b5
    update_ema = tf.group(update1,update2,update3,update4)
    return (h5,update_ema)
#======================函数定义结束==============================


#=======================构建Graph===============================

X = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])
is_test = tf.placeholder(tf.bool)
itr = tf.placeholder(tf.int32)
keep_drop = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
beta = tf.placeholder(tf.float32)
#tf_x_va = tf.constant(X_va)
#tf_x_te = tf.constant(X_test)

h,update_ema = Mode(X)

#regulization = beta*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W5)+tf.nn.l2_loss(W6)+tf.nn.l2_loss(W8))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h,labels=y))
#opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

predict = tf.nn.softmax(h)

with tf.Session() as sess:
    print('开始初始化......')
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 50
    step_num = 50000
    total = X_train.shape[0]
    avg_l = 0
    index_list = range(0,X_train.shape[0])

    for step in range(step_num):
        offset = (step*batch_size)%(total-batch_size)
        x_ = X_train[offset:offset+batch_size,:,:,:]
        y_ = y_train[offset:offset+batch_size,:]
        #执行train的时候，仅仅是生成新的mean和var并使用
       
        #------cal learning-rate--------
        max_learning_rate = 0.02
        min_learning_rate = 0.0005
        decay_speed = 1600
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-step/decay_speed)
        #lr = 0.001
        #-------------end--------------
        
        #feed_dict
        feed_dict_train = {X:x_,y:y_,itr:step*batch_size,is_test:False,keep_drop:0.5,learning_rate:lr}
        #feed_dict_test = {X:X_test,itr:step*batch_size,is_test:True,keep_drop:1,learning_rate:lr}
        
        #run bp
        _, l, pre_tr = sess.run([opt, loss, predict], feed_dict=feed_dict_train) #训练
        sess.run(update_ema, feed_dict=feed_dict_train)
        
        #cal avg_loss
        avg_l = avg_l + l/1000.0;
        if ( step%1000==0 ):
            #loss
            print('learning rate : %.4f' % lr)
            print('step : %d\naverage_loss : %f' % (step,avg_l))
            avg_l = 0
            #mini-batch
            print('Mini-batch accuracy : %.2f%%' % accuracy(pre_tr,y_))
            #sub-training set
            idx_list = sample(index_list,7000)
            sub_x_train = X_train[idx_list,:]
            sub_y_train = y_train[idx_list,:]
            feed_dict_test_train = {X:sub_x_train, itr:step*batch_size, is_test:True,keep_drop:1,learning_rate:lr}
            pre_tr8 = sess.run(predict, feed_dict=feed_dict_test_train)
            print('Random 7000 train accuracy : %.2f%%' % accuracy(pre_tr8, sub_y_train))
            #validation
            '''
            feed_dict_va = {X:x_va,itr:step*batch_size,is_test:True,keep_drop:1,learning_rate:lr}
            pre_va = sess.run(predict,feed_dict=feed_dict_va)
            print('validation accuracy : %.2f%%\n' % accuracy(pre_va,y_va))
            '''

    #print('validation accuracy : %.2f%%\n' % accuracy(pre_va,y_va))
    #pre_te = sess.run(predict,feed_dict=feed_dict_test)
    #print('test accuracy : %.2f%%\n' % accuracy(pre_te,y_test))

    #===========================生成文件=======================================

    test = pd.read_csv('test.csv')
    test = np.array(test)
    test = test.reshape((-1,28,28,1))
    t1 = test[:7000,:,:,:]
    t2 = test[7000:14000,:,:,:]
    t3 = test[14000:21000,:,:,:]
    t4 = test[21000:28000,:,:,:]

    p1 = sess.run(predict,feed_dict={X:t1,itr:16000,is_test:True,keep_drop:1})
    p2 = sess.run(predict,feed_dict={X:t2,itr:16000,is_test:True,keep_drop:1})
    p3 = sess.run(predict,feed_dict={X:t3,itr:16000,is_test:True,keep_drop:1})
    p4 = sess.run(predict,feed_dict={X:t4,itr:16000,is_test:True,keep_drop:1})

    pre1 = np.argmax(p1,1)
    pre2 = np.argmax(p2,1)
    pre3 = np.argmax(p3,1)
    pre4 = np.argmax(p4,1)

    ans = np.append(np.append(pre1,pre2),np.append(pre3,pre4))
    asub =  pd.DataFrame({'ImageId':list(range(1,28001)),'Label':np.int32(ans)})
    asub.to_csv('predict.csv',index=None)
        
