from torch.utils.data import Dataset
import PIL.Image as Image
import os
import torch
import numpy as np
from collections import Counter

def make_dataset_whole(root):
    # root = "./data/train"
    imgs = []
    for img_path in os.listdir(root):
        imgs.append(os.path.join(root,img_path))
    return imgs

def get_npy(ground_npy,dst_class_list):

    full_classname_list = ['jyxw', 'kzxg', 'mgas', 'tumor', 'yzxb']
    full_classname_list = ['jyxw', 'yzxb', 'zfbx', 'szbx']
    new_npy = np.ones((len(dst_class_list)+1,256,256))*len(dst_class_list)
    for classname_index in range(len(dst_class_list)):
        new_npy[classname_index] = ground_npy[full_classname_list.index(dst_class_list[classname_index])]
    zero_npy = np.zeros((256,256))
    for classname_index in range(len(dst_class_list)):
        zero_npy = zero_npy+new_npy[classname_index]
    new_npy[len(dst_class_list)] = np.where(zero_npy == 0 ,1, 0)
    return new_npy

def get_npy2(ground_npy,dst_class_list):

    full_classname_list = ['jyxw', 'kzxg', 'mgas', 'tumor', 'yzxb']
    src_classname_list = ['kzxg', 'mgas', 'tumor']
    src_classname_list = ['jyxw', 'yzxb', 'zfbx', 'szbx']
    full_classname_list = ['unlabel', 'jyxw', 'yzxb', 'zfbx', 'szbx']
    new_npy = np.zeros((len(dst_class_list)+1,256,256))
    for classname_index in range(len(dst_class_list)):
        src_index = src_classname_list.index(dst_class_list[classname_index])
        dst_index = full_classname_list.index(dst_class_list[classname_index])
        new_npy[dst_index] = ground_npy[src_index]
    zero_npy = np.zeros((256,256))
    for classname_index in range(1,len(dst_class_list)+1):
        zero_npy = zero_npy+new_npy[classname_index]
    new_npy[0] = np.where(zero_npy == 0 ,1, 0)
    return new_npy

class LiverDataset_whole(Dataset):
    def __init__(self, root,normalize=True):
        imgs = make_dataset_whole(root)
        self.imgs = imgs
        self.normalize = normalize
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data_path = self.imgs[index]
        img = Image.open(data_path).convert('RGB')
        img = np.array(img, dtype=np.float32)
        if self.normalize:
            img = (img - 128.0)/128.0
        img = np.transpose(img, [2, 0, 1])
        img_name = os.path.basename(data_path)
        return img,img_name