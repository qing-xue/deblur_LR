from random import shuffle
import glob
import numpy as np
import h5py
import sys
from PIL import Image

# REDS = 'F:/workplace/public_dataset/REDS'
# python im2hdf5.py F:/workplace/public_dataset/REDS /*/*.png 1000

if __name__ == '__main__':
    # REDS = 'F:/workplace/public_dataset/REDS'
    REDS = sys.argv[1]
    files = sys.argv[2]
    num_per_hf5 = int(sys.argv[3])    # notice that if it can divide exactly!!!

    train_blur_bicubic = REDS + '/train/train_blur_bicubic/X4'
    train_sharp = REDS + '/train/train_sharp'

    # read addresses and labels from the 'train' folder
    LR_train_path = train_blur_bicubic + files
    train_addrs = glob.glob(LR_train_path)

    # loop over train addresses
    train_x, train_sub4y, train_y = [], [], []
    for i in range(len(train_addrs)):
        # read an image and ...
        addr = train_addrs[i]
        ref_file = addr.replace("train_blur_bicubic/X4", "train_sharp")
        im_lr = Image.open(addr)
        im_sr = Image.open(ref_file)
        im_sub4x = im_sr.resize((im_sr.size[0] // 4, im_sr.size[1] // 4), Image.BICUBIC)
        train_x.append(np.array(im_lr))
        train_y.append(np.array(im_sr))
        train_sub4y.append(np.array(im_sub4x))

        if (i + 1) % num_per_hf5 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(train_addrs)))
            # open a hdf5 file and create earrays
            hdf5_path = train_blur_bicubic + '/dataset{}.hdf5'.format(i // num_per_hf5)
            hdf5_file = h5py.File(hdf5_path, 'w')
            dset1 = hdf5_file.create_dataset("data",  data = np.array(train_x).transpose((0, 3, 1, 2)))
            dset3 = hdf5_file.create_dataset("label", data = np.array(train_y).transpose((0, 3, 1, 2)))
            dset2 = hdf5_file.create_dataset("label_db", data = np.array(train_sub4y).transpose((0, 3, 1, 2)))

            train_x.clear()
            train_y.clear()
            train_sub4y.clear()
            hdf5_file.close()

    if  len(train_x) > 1:
        print('Train data: res/{}'.format(len(train_addrs)))
        # open a hdf5 file and create earrays
        hdf5_path = train_blur_bicubic + '/dataset_.hdf5'
        hdf5_file = h5py.File(hdf5_path, 'w')
        dset1 = hdf5_file.create_dataset("data",  data = np.array(train_x).transpose((0, 3, 1, 2)))
        dset3 = hdf5_file.create_dataset("label", data = np.array(train_y).transpose((0, 3, 1, 2)))
        dset2 = hdf5_file.create_dataset("label_db", data = np.array(train_sub4y).transpose((0, 3, 1, 2)))

