import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import numpy as np
import random
from os.path import join
import glob
import h5py
import sys
import os
from os.path import join


def is_image_file(filename):
    if filename.startswith('._'):
        return None
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DataValSet(data.Dataset):
    """ 获取验证集 LR_HR 配对数据 """

    def __init__(self, root_dir):
        self.input_dir = join(root_dir, r'val_blur_bicubic\X4')
        self.sr_dir = join(root_dir, 'val_sharp')
        self.input_names = []
        self.hr_names = []

        # 根据数据集已有目录结构获取，有耦合
        for sub_name in sorted(os.listdir(self.input_dir)):
            sub_dir = join(self.input_dir, sub_name)
            for img_name in sorted(os.listdir(sub_dir)):
                if not is_image_file(img_name): continue
                self.input_names.append(join(sub_name, img_name))
                self.hr_names.append(join(sub_name, img_name))

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, index):
        """ 前面先保存图片路径和文件名，仅在此处返回图像 """
        inputx = imread(join(self.input_dir, self.input_names[index])).transpose((2, 0, 1))
        inputx = np.asarray(inputx, np.float32).copy() / 255
        target = imread(join(self.sr_dir, self.hr_names[index])).transpose((2, 0, 1))
        target = np.asarray(target, np.float32).copy() / 255
        return inputx, target


class DataTrainSet(data.Dataset):
    """ 获取训练集 .hdf5 数据 """

    def __init__(self, h5py_file_path):
        super(DataSet, self).__init__()  
        self.hdf5_file  = h5py_file_path
        self.file    = h5py.File(self.hdf5_file, 'r')
        self.inputs  = self.file['data']
        self.deblurs = self.file.get("label_db")
        self.hrs     = self.file.get("label")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        # 这里需要检查范围是否为 [0,1]
        input_patch  = np.asarray(self.inputs[index, :, :, :], np.float32) / 255
        deblur_patch = np.asarray(self.deblurs[index, :, :, :], np.float32) / 255
        hr_patch     = np.asarray(self.hrs[index, :, :, :], np.float32) / 255
        # randomly flip
        if random.randint(0, 1) == 0:
            input_patch  = np.flip(input_patch, 2)
            deblur_patch = np.flip(deblur_patch, 2)
            hr_patch     = np.flip(hr_patch, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
        deblur_patch   = np.rot90(deblur_patch, rotation_times, (1, 2))
        hr_patch       = np.rot90(hr_patch, rotation_times, (1, 2))

        return input_patch.copy(),\
               deblur_patch.copy(),\
               hr_patch.copy()



