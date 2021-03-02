import os
import sys
import numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset
import random
from datasets.dataset_hf5 import is_image_file


# 作为验证集需改为顺序读取
def find_lr_hr_file2(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.png'):
                fullname = os.path.join(root, f)
                ref_full = fullname.replace("val_blur_bicubic/X4", "val_sharp")
                yield fullname, ref_full


# ugly... 作为验证集需改为顺序读取
def find_lr_hr_file(root, subs_path="val/val_blur_bicubic/X4"):
    for _dir in sorted(os.listdir(root)):
        child = os.path.join(root, _dir)
        if is_image_file(child):
            fullname = child
            ref_full = fullname.replace(subs_path, "val/val_sharp")
            # print(fullname)
            yield fullname, ref_full
        else:
            for x in sorted(os.listdir(child)):
                _file = os.path.join(child, x)
                if is_image_file(_file):
                    fullname = _file
                    ref_full = fullname.replace(subs_path, "val/val_sharp")
                    # print(fullname)
                    yield fullname, ref_full


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self, imgs_lr_path):
        super(Dataset, self).__init__()     # 仍继承父类原有属性，防止def init覆盖初始化
        self.imgs_lr_path  = imgs_lr_path

        train_x, train_y, train_sub4y = [], [], []
        for bulr_file, ref_file in find_lr_hr_file(self.imgs_lr_path):
            im_lr = Image.open(bulr_file)
            im_sr = Image.open(ref_file)
            im_sub4x = im_sr.resize((im_sr.size[0] // 4, im_sr.size[1] // 4), Image.BICUBIC)
            train_x.append(np.array(im_lr))
            train_y.append(np.array(im_sr))
            train_sub4y.append(np.array(im_sub4x))

        self.x_data = np.array(train_x).transpose((0, 3, 1, 2))
        self.y_data = np.array(train_y).transpose((0, 3, 1, 2))
        self.sub4y_data = np.array(train_sub4y).transpose((0, 3, 1, 2))
        self.len = len(train_x)
    
    def __getitem__(self, index):
        # preprocess
        input_patch  = np.asarray(self.x_data[index, :, :, :], np.float32) / 255
        hr_patch     = np.asarray(self.y_data[index, :, :, :], np.float32) / 255
        deblur_patch = np.asarray(self.sub4y_data[index, :, :, :], np.float32) / 255

        # mind the order!!!
        return input_patch.copy(),\
               hr_patch.copy(),\
               deblur_patch.copy()

    def __len__(self):
        return self.len


class TripleDataSet(Dataset):
    def __init__(self, h5py_file_path):
        super(Dataset, self).__init__()  
        self.hdf5_file  = h5py_file_path

        self.file    = h5py.File(self.hdf5_file, 'r')
        # print(self.file.keys())
        self.inputs  = self.file.get("data")
        self.deblurs = self.file.get("label_db")
        self.hrs     = self.file.get("label")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        input_patch  = np.asarray(self.inputs[index, :, :, :], np.float32) / 255
        deblur_patch = np.asarray(self.deblurs[index, :, :, :], np.float32) / 255
        hr_patch     = np.asarray(self.hrs[index, :, :, :], np.float32) / 255
        
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
        deblur_patch   = np.rot90(deblur_patch, rotation_times, (1, 2))
        hr_patch       = np.rot90(hr_patch, rotation_times, (1, 2))

        return input_patch.copy(),\
               deblur_patch.copy(),\
               hr_patch.copy()

if __name__ == '__main__':
    # datadir = sys.argv[1]
    # print(datadir)
    # dealDataset = DealDataset(datadir)

    datadir = 'F:/workplace/public_dataset/REDS/train/h5/dataset0.hdf5'
    tripleDataSet = TripleDataSet(datadir)
    pass

# python GFN/datasets/my_dataset.py F:/workplace/public_dataset/REDS/train/train_blur_bicubic/X4/000
# python GFN/datasets/my_dataset.py F:/workplace/public_dataset/REDS/train/train_blur_bicubic/X4/dataset10.hdf5