from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras.models import Model, Sequential
from PIL import Image
import numpy as np
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']='2'
import time
#模型使用参考https://blog.csdn.net/qq_37588821/article/details/97174536
folders = './data/'
time.sleep(10)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
out = GlobalAveragePooling2D()(base_model.output)
model = Model(base_model.input, out)
with open('data.csv') as f:
    df = pd.read_csv(f, delimiter=',', header=None)
name1 = df[1:512][0].astype(str)
# print name1
# print name2
name=name1+'.jpg'
data = []
k=0
for sub_folder in name:
    data_frame = []
    frame_filename = folders + sub_folder
    frame_file = Image.open(frame_filename)
    frame_file= frame_file.convert('RGB')
    frame_file = frame_file.resize((224, 224))
    frame_file_arr = np.array(frame_file)
    frame_file_arr=np.expand_dims(frame_file_arr,axis=0)
    # data_frame.append(frame_file_arr)
    # data_frame = np.array(data_frame)
    # expanded_data = np.tile(frame_file_arr[np.newaxis,:,:,:], (100, 1,1,1))
    # expanded_data = np.repeat(frame_file_arr[:,:,np.newaxis],100,axis=2)
    model_output = model.predict(frame_file_arr)
    k=k+1
    print (k)

    data.append(model_output)
data = np.array(data)
np.save('picture_feature_test.npy', data)



"""
import os
from PIL import Image
import jieba
import re
path = 'data'
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x[:-4]))
#print(path_list)
txt_origin=[]
jpg_origin=[]
string = "~!@#$%^&*()_+-*/<>,.[]\/"
for i in path_list:
    if ".jpg" in i:
        img_PIL = Image.open("data/"+i)#读取数据
        img_PIL = np.array(img_PIL)
        jpg_origin.append(img_PIL)
    if ".txt" in i:
        file = open('data/'+i, 'r',encoding="utf-8",errors = 'ignore')     # 打开文件
        data = file.read()
        data = re.sub('\W+', ' ', data).replace("_", ' ')
        data=list(jieba.cut_for_search(data))
        data=[x.strip() for x in data if x.strip() != '']
        txt_origin.append(data)  
"""