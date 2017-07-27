from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import pandas as pd
import tensorflow as tf

pickle_file = 'notMNIST.pickle'

#load data 
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

#reformat data
image_size = 28
num_labels = 10
num_channels = 1 # grayscale
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return (dataset, labels)
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

#========================parameter set====================

hidden1_depth = 16
hidden2_depth = 32
hidden3_num = 256
hidden4_num = 128

filter1_size = 5
filter2_size = 5

#==========================Set end===========================


#========================variable set=========================


#graph = tf.Graph()
#with graph.as_default():
#xavier初始化
W_1=tf.Variable(tf.truncated_normal([filter1_size,filter1_size,num_channels,hidden1_depth],stddev=0.1))
W_2=tf.Variable(tf.truncated_normal([filter2_size,filter2_size,hidden1_depth,hidden2_depth],stddev=0.1))
W_3=tf.Variable(tf.truncated_normal([(image_size//4)*(image_size//4)*hidden2_depth, hidden3_num],stddev=0.1))
W_4=tf.Variable(tf.truncated_normal([hidden3_num, num_labels],stddev=0.1))
#W_5=tf.Variable(tf.truncated_normal([hidden4_num, num_labels],stddev=0.1))

b_1=tf.Variable(tf.zeros([hidden1_depth])+1)
b_2=tf.Variable(tf.zeros([hidden2_depth])+1)
b_3=tf.Variable(tf.zeros([hidden3_num])+1)
b_4=tf.Variable(tf.zeros([num_labels])+1)
#b_5=tf.Variable(tf.zeros([num_labels])+1)

#==========================Set end=============================

#========================wrapper function=========================


def Mode(X):
	conv_1 = tf.nn.conv2d(X,W_1,strides=[1,1,1,1],padding='SAME') 
	hidden_1 = tf.nn.elu(tf.add(conv_1,b_1))
	pool_1 = tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	#lay2
	conv_2 = tf.nn.conv2d(pool_1,W_2,[1,1,1,1],padding='SAME') 
	hidden_2 = tf.nn.elu(tf.add(conv_2,b_2))
	pool_2 = tf.nn.max_pool(hidden_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	#lay3
	Shape = tf.reshape(pool_2,shape=(-1,W_3.get_shape().as_list()[0]))
	hidden_3 = tf.add(tf.matmul(Shape,W_3),b_3)
	lay_3 = tf.nn.elu(hidden_3)
	#drop_3 = tf.nn.dropout(lay_3,0.5)
	
	#lay4
	#hidden_4 = tf.nn.elu(tf.matmul(drop_3,W_4)+b_4)
	#dropout_4 = tf.nn.dropout(hidden_4,0.5)
	
	output = tf.matmul(hidden_3,W_4)+b_4
	return output


#============================Set end==============================


X = tf.placeholder(tf.float32,[None,image_size,image_size,num_channels])
y = tf.placeholder(tf.float32,[None,num_labels])

#=========================WARNING!!========================
tf_valid_dataset = tf.constant(valid_dataset)
tf_test_dataset = tf.constant(test_dataset)
#=========================WARNING!!========================

h = Mode(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h,labels=y))
learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

train_predict = tf.nn.softmax(h)
vaild_predict = tf.nn.softmax(Mode(tf_valid_dataset))
test_predict = tf.nn.softmax(Mode(tf_test_dataset))


#=========================training it==========================

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	batch_size = 128
	num_step = 53001
	avg_l=0
	for step in range(num_step):
		offset = (step*batch_size)%(train_dataset.shape[0]-batch_size)
		xx_ = train_dataset[offset:(offset+batch_size),:,:,:]
		yy_ = train_labels[offset:(offset+batch_size),:]

		feed_ = {X:xx_,y:yy_}
		_,l,predict = sess.run([opt,loss,train_predict],feed_dict=feed_)
		avg_l = avg_l + l/500.0 
		if (step%500==0):
			print("%d steps , avg-Loss = %f" % (step,avg_l))
			avg_l = 0
			print("Mini-batch accuracy : %.1f%%" % accuracy(predict,yy_) )
			print("Validation accuracy : %.1f%%" % accuracy(sess.run(vaild_predict),valid_labels) )
			
	print("\n\nTest Accuracy = %.2f%%" % accuracy(sess.run(test_predict),test_labels))
	print("Validation Accuracy = %.2f%%" % accuracy(sess.run(vaild_predict),valid_labels))
	
#===========================End=====================================

