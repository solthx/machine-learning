from keras import applications
from keras.layers import *  
from keras.models import *
from keras.preprocessing.image import *
import h5py
import preprocessing

def models_generate_data(MODEL,Shape,MODEL_name,pre_func=None):
    input_tensor = Input((Shape[0],Shape[1],3))
    x = input_tensor
    if pre_func is not None:
        x = Lambda(pre_func)(input_tensor)  
    base_model = MODEL(include_top=False,input_tensor=x)
    #由于在全连接层的参数太多, 所以使用globalaveragepooling
    output_ = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input,outputs=output_)

    #接下来是把dir里的img都规则的读进来，获得图片的batch生成器
    a = None
    if MODEL_name[:3]=='vgg':
        a = applications.vgg16.preprocess_input
    img_batch_generator = ImageDataGenerator()
    #这里注意！flow_from_directory会从这个路径下找 文！件！夹！
    #并且把文件夹下的图片当作一类！
    #如果找不到文件夹，n直接就是0了！ 然后就bug了！！ （调了半天T_T

    train_img_generator = img_batch_generator.flow_from_directory('train_img',target_size=Shape,shuffle=False,batch_size=25)
    test_img_generator = img_batch_generator.flow_from_directory('test',target_size=Shape,shuffle=False,class_mode=None,batch_size=25)

    #之后就可以predict了
    #step = n/batch_size
    train = model.predict_generator(train_img_generator,1000,verbose=1)
    test = model.predict_generator(test_img_generator,500,verbose=1)

    #保存预测值
    with h5py.File(MODEL_name+'.h5') as h:
        h.create_dataset('train',data=train)
        h.create_dataset('labels',data=train_img_generator.classes)
        h.create_dataset('test',data=test)

preprocessing.preprocess()
#models_generate_data(applications.InceptionV3,(299,299),'InceptionV3',applications.inception_v3.preprocess_input)
#models_generate_data(applications.ResNet50,(224,224),'ResNet50')
#models_generate_data(applications.Xception,(299,299),'Xception',applications.xception.preprocess_input)
models_generate_data(applications.vgg16.VGG16,(224,224),'vgg16')
applications.vgg16.preprocess_input()
