import tensorflow as tf
from tensorflow import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.callbacks import TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adam

def build_model(base_model_name,weights):
    if base_model_name=="VGG16":
        base_model=VGG16(weights=weights,include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(43, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

batch_size=32
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        fill_mode="reflect")

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'data/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
experiment_index="experiment2"
tensorboard=TensorBoard(log_dir='./logs/'+experiment_index, update_freq=500)
reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=1, min_lr=1e-6)
model=build_model("VGG16","imagenet")
model.compile(Adam(1e-4), loss="categorical_crossentropy", metrics=["acc"])
model.fit_generator(train_generator,epochs=40,validation_data=validation_generator,workers=4,callbacks=[tensorboard,reduce_lr])
if not os.path.exists("weights"+os.sep+experiment_index):
    os.mkdir("weights"+os.sep+experiment_index)
model.save("weights"+os.sep+experiment_index+os.sep+"last_model.h5")
