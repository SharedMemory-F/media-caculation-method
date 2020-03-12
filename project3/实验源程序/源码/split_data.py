import os
import random
import shutil
class_dir=os.listdir("50_ObjectCategories")
for class_name in class_dir:
    train_path="data/train"+os.sep+class_name
    val_path="data/val"+os.sep+class_name
    insider_path="data/insider"+os.sep+class_name

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    if not os.path.exists(insider_path):
        os.mkdir(insider_path)
    class_path="50_ObjectCategories"+os.sep+class_name
    img_names=os.listdir(class_path)
    for img_name in img_names[:1]:
        img_path=class_path+os.sep+img_name
        if os.path.isfile(img_path):
            shutil.copy(img_path,insider_path+os.sep+img_name)
            
for class_name in class_dir:
    train_path="data/train"+os.sep+class_name
    val_path="data/val"+os.sep+class_name
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    class_path="50_ObjectCategories"+os.sep+class_name
    img_names=os.listdir(class_path)
    for img_name in img_names:
        img_path=class_path+os.sep+img_name
        if os.path.isfile(img_path):
            if random.random()>0.8:
                shutil.copy(img_path,val_path+os.sep+img_name)
            else:
                shutil.copy(img_path,train_path+os.sep+img_name)