1.代码运行：

训练模型和预测：只需运行pro5.ipynb文件即可,但需要下载所有文件，尤其是预先保存的特征矩阵的三个npy文件，其中的normalized_data因为超过上载的25mb大小，传在tag的release文件中。另外只能运行一次(由于变量名和数量设置会因为数量超出和名字重复报错)，如需再次重新训练，需要重启内核清除内存数据


2.提交预测结果：result.txt中


3.实验报告：实验报告.pdf中


4.其他文件标注：

        data.cvs整理为0,1,-1的训练集标签数据
        
        picturevgg16.py 使用keras-vgg16对图片数据进行特征提取
        
        data文件夹、train_without_label.txt、train.txt为提供的最初数据
        
        picture_feature_train.np和picture_feature_test.np为提取特征后存贮的图片特征numpy矩阵
        
        normalized_data.py为文本归一向量化后的numpy矩阵
        
        keras_attention_mechanism_model.py为注意力机制
        
