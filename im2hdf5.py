from random import shuffle
import glob
import numpy as np
import h5py
import sys
from PIL import Image
import os
from matplotlib import pyplot as plt
import random

def im_crop(im, box_w=256, box_h=256, stride=256):
    width = im.size[0]
    height = im.size[1]
    patches = []

    iw = np.arange(0, width  - box_w + 1, stride)
    jh = np.arange(0, height - box_h + 1, stride)
    for i in iw:
        for j in jh:
            cm = im.crop(box = (i, j, i + box_w, j + box_h))
            # print((i, j, i + box_w, j + box_h))
            patches.append(cm) 

    # 方案一：补边 方案二：两头向中间（waiting）
    if width % box_w != 0:
        for j in jh:
            cm = im.crop(box = (width - box_w, j, width, j + box_h))
            # print((width - box_w, j, width, j + box_h))
            patches.append(cm) 
        if height % box_h != 0:
            # 可能与另一方向的有重复
            cm = im.crop(box = (width - box_w, height - box_h, width, height))
            patches.append(cm) 
    if height % box_h != 0:
        for i in iw:
            cm = im.crop(box = (i, height - box_h, i + box_w, height))
            # print((i, height - box_h, i + box_w, height))
            patches.append(cm) 
        if width % box_w != 0:
            cm = im.crop(box = (width - box_w, height - box_h, width, height))
            patches.append(cm) 

    return patches

def test_im_crop():
    im = Image.open('F:/workplace/public_dataset/REDS/val/val_sharp/000/00000000.png')
    patches = im_crop(im)
    for i in patches:
        i.show()

def im_LRHR_show(im_lr, im_hr, im_lr_GT):
    plt.subplot(2,2,1)
    plt.imshow(im_lr, cmap='gray')
    plt.title('im_lr')
    plt.subplot(2,2,2)
    plt.imshow(im_lr_GT, cmap='gray')
    plt.title('im_lr_GT')
    plt.subplot(2,2,3)
    plt.imshow(im_hr, cmap='gray')
    plt.title('im_hr')
    plt.show()
    plt.close()


# python im2hdf5.py F:/workplace/public_dataset/REDS /*/*.png 1000
# REDS = 'F:/workplace/public_dataset/REDS'
# files = '/*/*.png'
# num_per_hf5 = 1000  
if __name__ == '__main__':

    REDS = 'F:/workplace/public_dataset/REDS'
    files = '/*/*.png'
    num_per_hf5 = 100 
    # REDS = sys.argv[1]
    # files = sys.argv[2]
    # num_per_hf5 = int(sys.argv[3])    # notice that if it can divide exactly!!!

    train_blur_bicubic = REDS + '/train/train_blur_bicubic/X4'
    train_sharp = REDS + '/train/train_sharp'

    # read addresses and labels from the 'train' folder
    LR_train_path = train_blur_bicubic + files
    train_addrs = glob.glob(LR_train_path)
    random.shuffle(train_addrs)

    # loop over train addresses
    h5_folder = REDS + '/train/h5'
    if not os.path.exists(h5_folder):
        os.makedirs(h5_folder)
    train_x, train_sub4y, train_y = [], [], []
    for i in range(len(train_addrs)):
        # read an image and pre-process...
        addr = train_addrs[i]
        ref_file = addr.replace("train_blur_bicubic/X4", "train_sharp")
        im_lr = Image.open(addr)
        im_sr = Image.open(ref_file)
        im_sub4y = im_sr.resize((im_sr.size[0] // 4, im_sr.size[1] // 4), Image.BICUBIC)
  
        # get patches
        im_lr_patches = im_crop(im_lr, box_w=256//4, box_h=256//4, stride=256//4)
        im_sr_patches = im_crop(im_sr, box_w=256, box_h=256, stride=256)
        im_sub4y_patches = im_crop(im_sub4y, box_w=256//4, box_h=256//4, stride=256//4)
        for each in zip(im_lr_patches, im_sr_patches, im_sub4y_patches):
            # im_LRHR_show(each[0], each[1], each[2])
            train_x.append(np.array(each[0]))
            train_y.append(np.array(each[1]))
            train_sub4y.append(np.array(each[2]))

        # 分批次保存
        if (i + 1) % num_per_hf5 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(train_addrs)))
            # open a hdf5 file and create earrays
            hdf5_path = h5_folder + '/dataset{}.hdf5'.format(i // num_per_hf5)
            hdf5_file = h5py.File(hdf5_path, 'w')
            dset1 = hdf5_file.create_dataset("data",  data = np.array(train_x).transpose((0, 3, 1, 2)))
            dset3 = hdf5_file.create_dataset("label", data = np.array(train_y).transpose((0, 3, 1, 2)))
            dset2 = hdf5_file.create_dataset("label_db", data = np.array(train_sub4y).transpose((0, 3, 1, 2)))
            train_x.clear()
            train_y.clear()
            train_sub4y.clear()
            hdf5_file.close()

    # 若还有剩余
    if  len(train_x) > 1:
        print('Train data: res/{}'.format(len(train_addrs)))
        # open a hdf5 file and create earrays
        hdf5_path = h5_folder + '/dataset_.hdf5'
        hdf5_file = h5py.File(hdf5_path, 'w')
        dset1 = hdf5_file.create_dataset("data",  data = np.array(train_x).transpose((0, 3, 1, 2)))
        dset3 = hdf5_file.create_dataset("label", data = np.array(train_y).transpose((0, 3, 1, 2)))
        dset2 = hdf5_file.create_dataset("label_db", data = np.array(train_sub4y).transpose((0, 3, 1, 2)))

