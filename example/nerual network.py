import pandas as pd
import numpy as np
import tensorflow as tf


#转换成one-hot
def Trans(y):
    m = y.shape[0]
    mat = np.zeros((m,10))
    for i in range(m):
        mat[i][y[i]] = 1
    return mat

#输出准确率
def accuracy(x,y):
	return (100.0*np.sum(np.argmax(x,1)== np.argmax(y,1)) / y.shape[0])
	

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

#隐藏层的节点数
hidden_nodes = 1024


#lay1
W1 = tf.Variable(tf.truncated_normal([28*28,hidden_nodes]))
b1 = tf.Variable(tf.zeros([hidden_nodes]))
#在第一层上对训练集、验证集、测试集进行预测
tr_lay_1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
va_lay_1 = tf.nn.relu(tf.add(tf.matmul(xx_va,W1),b1))
te_lay_1 = tf.nn.relu(tf.add(tf.matmul(xx_test,W1),b1))

#lay2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes,10]))
b2 = tf.Variable(tf.zeros([10]))
#假设函数
h = tf.add(tf.matmul(tr_lay_1,W2),b2)
#在第二层上对训练集、验证集、测试集进行预测
tr_pre = tf.nn.softmax(h)
va_pre = tf.nn.softmax(tf.matmul(va_lay_1,W2)+b2)
te_pre = tf.nn.softmax(tf.matmul(te_lay_1,W2)+b2)


#用相对熵作为Loss
Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=h))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(Loss)
init = tf.global_variables_initializer()


#mini-batch gradient descent
with tf.Session() as sess:
	sess.run(init)
	
	#设置超参数
	batch_size = 128
	num_step = 3001
	learning_rate = 0.1
	
	for i in range(num_step):
		offset = (i*batch_size)%(xx_train.shape[0]-batch_size)
		x_ = xx_train[offset:(offset+batch_size),:]
		y_ = y_train[offset:(offset+batch_size),:]
		_,l,predict = sess.run([train_step,Loss,tr_pre],feed_dict={X:x_,y:y_})
		if ( i%500==0 ):
			print("Minibatch loss at step %d:%f" % (i,l))
			print("mini-batch accuracy: %.1f%%" % accuracy(predict,y_))
			print("validation accuracy: %.1f%%" % accuracy(sess.run(va_pre),y_va))
			
	#整体评估
	correct_predict = tf.equal(tf.argmax(h,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
	print('\ntrain Accuracy =',accuracy.eval(session=sess,feed_dict={X:xx_train,y:y_train}))
	print('\ntest Accuracy =',accuracy.eval(session=sess,feed_dict={X:xx_test,y:y_test}))
	print('\nva Accuracy =',accuracy.eval(session=sess,feed_dict={X:xx_va,y:y_va}))



