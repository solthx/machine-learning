import numpy as np
import os
from os import walk
import shutil

def preprocess():
    if os.path.exists('train_img'):
        print('train_img has existed!')
        return None
    os.makedirs('train_img/dog')
    os.makedirs('train_img/cat')
    #if not os.path.exists('test_img'):
    #    os.mkdir('test_img')
    dir_list = os.listdir('train')
    dog_img_name = filter(lambda x:x[:3]=='dog',dir_list)
    cat_img_name = filter(lambda x:x[:3]=='cat',dir_list)
    #复制
    print('开始复制...')
    for filename in dog_img_name:
        shutil.copy('./train/'+filename,'./train_img/dog/'+filename)
    for filename in cat_img_name:
        shutil.copy('./train/'+filename,'./train_img/cat/'+filename)
    print('复制结束...')

