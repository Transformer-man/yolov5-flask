import os,shutil
import numpy as np
import cv2
from tqdm import tqdm
#上一步保存的所有image和label文件路径
image_root = r'../datasets/yolo_data/images'
label_root = r'../datasets/yolo_data/labels'
names = []
for root,dir,files in os.walk(label_root ):
    for file in files:
        names.append(file)
val_split = 0.1
np.random.seed(10101)
np.random.shuffle(names)
num_val = int(len(names)*val_split)
num_train = len(names) - num_val
trains = names[:num_train]
vals = names[num_train:]
#保存路径
save_path_img = r'../datasets//traindata'
if not os.path.exists(save_path_img):
    os.mkdir(save_path_img)
def get_train_val_data(img_root,txt_root,save_path_img,files,typ):
        def get_path(root_path,path1):
            path = os.path.join(root_path,path1)
            if not os.path.exists(path):
                os.mkdir(path)
            return path
        for val in tqdm(files):
            txt_path = os.path.join(txt_root,val)
            img_path = os.path.join(img_root,val.split('.')[0]+'.jpg')
            img_path1 = get_path(save_path_img,'images')
            txt_path1 = get_path(save_path_img,'labels')
            rt_img = get_path(img_path1,typ)
            rt_txt = get_path(txt_path1,typ)
            txt_path1 = os.path.join(rt_txt,val)
            img_path1 = os.path.join(rt_img,val.split('.')[0]+'.jpg')
            shutil.copyfile(img_path, img_path1)
            shutil.copyfile(txt_path,txt_path1)
get_train_val_data(image_root,label_root,save_path_img,vals,'val')
get_train_val_data(image_root,label_root,save_path_img,trains,'train') 

    
    
    
    
    