import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
	
	
#--------数据预处理---------
dataset = pd.read_csv('breast.csv',header=None)	   		
dataset = dataset.replace(to_replace='?',value=np.nan)	
dataset = dataset.dropna()
Row = np.arange(1,10)
X_train,X_test,y_train,y_test = train_test_split(dataset[Row],dataset[10],random_state=40,test_size=0.25)
y_train = y_train.replace(to_replace=[4,2],value=[1,0])
y_test = y_test.replace(to_replace=[4,2],value=[1,0])
y_train = np.reshape(y_train,(y_train.shape[0],1))
y_test = np.reshape(y_test,(y_test.shape[0],1))

#特征数和训练样本数
feature_num = X_train.shape[1]
train_num = X_train.shape[0]


#占位符
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


#注意！权值的初始化应该全为0
W = tf.Variable(tf.zeros([feature_num,1]))
b = tf.Variable(-.9)

#假设函数
h = tf.sigmoid(tf.matmul(X,tf.reshape(W, [-1, 1]))+b)

c1 = y * tf.log(h)
c2 = (1 - y) * tf.log(1 - h)
cost = -(c1 + c2) / train_num 
loss = tf.reduce_sum(cost)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
	sess = tf.Session()
	sess.run(init)

	for i in range(10000):
		sess.run(train_step,{X:X_train,y:y_train})
		#if i%100000==0:
		#	print('W =',sess.run(W),'b =',sess.run(b))

	X_test = X_test.astype('float')
	w = sess.run(W)
	b = sess.run(b)
	hy = tf.sigmoid(X_test.dot(w)+b)
	hy = sess.run(hy)
	ans = (hy>0.5).astype(int)-y_test
	ans = sum((ans==0).astype(int))
	print('My Accuracy :',ans/y_test.shape[0])

	from sklearn.linear_model import LogisticRegression
	lr = LogisticRegression()
	lr.fit(X_train,y_train)
	print('sklearn_Logistic_score ：',lr.score(X_test,y_test))

	
