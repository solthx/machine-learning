import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras import optimizers,initializers,losses
import pandas as pd

#此时的train其实已经是分类好的了
#但我们并不知道是怎么分的类
#但我们有labels就是这里的y_train
#因此我们可以利用这个labels
#再加一层输出层，从2048→1 最小化loss来得到最后的猫狗分类
#因为train已经是分类好的了，所以很快就会很清晰的分开了

X_train = []
X_test = []


#load data
name_list = ['InceptionV3.h5','ResNet50.h5','Xception.h5']
for filename in name_list:
    with h5py.File(filename,'r') as h:
        X_train.append(np.array(h['train']))
        y_train=np.array(h['labels'])
        X_test.append(np.array(h['test']))

#0是垂直排，1是水平排
#水平排过之后，种类变成了三倍
#所以把三个模型的预测综合一下，用这个大集合来预测

X_train = np.concatenate(X_train,axis=1)
X_test = np.concatenate(X_test,axis=1)
#因为前面的训练可能会次数变多（batch_size*800）和(batch_size*500)
#X_train = X_train[:25000]

#construct graph
inputs = Input(X_train.shape[1:])
x = Dropout(0.5)(inputs) #防止过拟合
x = Dense(1,activation='sigmoid')(inputs)

model = Model(inputs,x)
model.compile(optimizers.Adam(),losses.binary_crossentropy,['accuracy'])
model.fit(X_train,y_train,batch_size=128,epochs=10,validation_split=0.2)

#predict
test_predict = model.predict(X_test,batch_size=32,verbose=1)
test_predict = test_predict.clip(min=0.05,max=0.995)

#low_from_directory得到的filenames是乱序的。。 所以还要特殊处理一下。。
test_img_generator = image.ImageDataGenerator()
nn = test_img_generator.flow_from_directory('test/',shuffle=False)

ans = np.zeros((12500,),dtype='float32')
#save
cnt = 0
for fname in nn.filenames:
    idx = fname.split('.')[0]
    idx = int(idx.split('\\')[-1])
    ans[idx-1] = test_predict[cnt]
    cnt+=1

ans = pd.DataFrame({'id':list(range(1,12501)),'label':ans})
ans.to_csv('1.csv',index=None)
