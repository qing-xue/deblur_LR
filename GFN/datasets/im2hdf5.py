from random import shuffle
import glob
import numpy as np
import h5py
import sys
from PIL import Image, ImageDraw
import os
from matplotlib import pyplot as plt
import random
import argparse

parser = argparse.ArgumentParser(description="Image to .hdf5 fills")
parser.add_argument("--REDS", required=True, help="REDS dataset dir, eg. 'dataset/REDS'")
parser.add_argument("--imgs", required=True, help="Image file to match, eg. '/*/*.png'")
parser.add_argument("--batch", default=1000, type=int, help="How many images per .hdf5")


def im_crop(im, box_w=256, box_h=256, stride_w=256, stride_h=256, epsilon=10):
    """Crop image to get patches.

    :param epsilon: 边界容忍值，低于之则直接丢弃
    :return: 返回截取的 patches 以及其对应于原图的坐标
    """
    width = im.size[0]
    height = im.size[1]
    patches, patches_idx = [], []
    iw = np.arange(0, width  - box_w + 1, stride_w)
    jh = np.arange(0, height - box_h + 1, stride_h)
    for i in iw:
        for j in jh:
            box = (i, j, i + box_w, j + box_h)
            cm = im.crop(box)
            patches.append(cm) 
            patches_idx.append(box)
    # repair x and y orientation's boundary
    if width - box_w - iw[-1] > epsilon:
        for j in jh:
            box = (width - box_w, j, width, j + box_h)
            cm = im.crop(box)
            patches.append(cm) 
            patches_idx.append(box)
    if height - box_h - jh[-1] > epsilon:
        for i in iw:
            box = (i, height - box_h, i + box_w, height)
            cm = im.crop(box)
            patches.append(cm) 
            patches_idx.append(box)
    # need only once
    if width - box_w - iw[-1] > epsilon and height - box_h - jh[-1] > epsilon:
        box = (width - box_w, height - box_h, width, height)
        cm = im.crop(box)
        patches.append(cm) 
        patches_idx.append(box)

    return patches, patches_idx


def test_im_crop():
    im = Image.open('F:/workplace/public_dataset/REDS/val/val_sharp/000/00000000.png')
    patches, boxes = im_crop(im, box_w=256, box_h=256, stride_w=256, stride_h=256)
    draw = ImageDraw.Draw(im)
    for box in boxes:
        color = (random.randint(64,255), random.randint(64,255), random.randint(64,255))
        draw.rectangle(box, outline=color, width=3)
        im.show()
    im.show()


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


# parser 命令行用法：= 号后面不加引号；空格后面加引号
# python im2hdf5.py --REDS=F:/workplace/public_dataset/REDS --imgs=/*/*.png --batch=1000
if __name__ == '__main__':
    opt = parser.parse_args()
    REDS = opt.REDS
    files = opt.imgs
    num_per_hf5 = opt.batch

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
        im_lr_patches, _ = im_crop(im_lr, box_w=256//4, box_h=256//4, stride_w=256//4, stride_h=256//4)
        im_sr_patches, _ = im_crop(im_sr, box_w=256, box_h=256, stride_w=256//4, stride_h=256//4)
        im_sub4y_patches, _ = im_crop(im_sub4y, box_w=256//4, box_h=256//4, stride_w=256//4, stride_h=256//4)
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

