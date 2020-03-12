#%% 
import tensorflow as tf
from tensorflow import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from matplotlib import pyplot as plt 
import numpy as np

def model_insider(model,layer_names):
    #把需要输出的层的名字加入layer_names 即可
    klayers=[model.get_layer(name).output for name in layer_names]
    insider_model=Model(inputs=model.input,outputs=klayers)
    return insider_model

def layer_to_img(layer):
    #把要可视化的层转化成图片，主要是进行层的二维化
    layer=np.squeeze(layer)
    if layer.ndim==1:
        #一维的层，在列方向上重复50次，宽度是50
        layer=np.repeat(np.array([layer]),50,axis=0)
        #print(layer.shape)
    elif layer.ndim==3:
        #三维的层进行拼接
        layer=np.reshape(layer,(layer.shape[0]*layer.shape[1],-1))
    return layer

if __name__ == "__main__":
    test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size=1
    validation_generator = test_datagen.flow_from_directory(
            'data/insider',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical')
    model=load_model("weights/experiment2/last_model.h5")
    print(model.summary())
    layer_names=["block5_pool","global_average_pooling2d_1","dense_1","dense_2"]
    insider_model=model_insider(model,layer_names)
    test_num = 0
    for batch in validation_generator:
        target_label=batch[1][0]
        input_img=batch[0]
        output=insider_model.predict(input_img)
        #plt.subplots_adjust(hspace=1.3, wspace=1.3)
        fig,axes=plt.subplots(len(layer_names)+1,1)
        fig.suptitle(layer_names)
        axes[0].imshow(input_img[0])#显示原图
        for i,layer_name in enumerate(layer_names):
            layer_img=layer_to_img(output[i])
            axes[i+1].imshow(layer_img)
            #axes[i+1].set_title(layer_name)
        test_num += 1
        result_path = "result"+os.sep+"test_"+str(test_num)+".jpg"
        fig.savefig(result_path)
# %%
