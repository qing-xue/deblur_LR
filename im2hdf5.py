from random import shuffle
import glob
import numpy as np
import tables
from PIL import Image

REDS = 'F:/workplace/public_dataset/REDS'
train_blur_bicubic = REDS + '/train/train_blur_bicubic/X4'
train_sharp = REDS + '/train/train_sharp'

# read addresses and labels from the 'train' folder
LR_train_path = train_blur_bicubic + '/*/*.png'
addrs = glob.glob(LR_train_path)

# 'th' for Theano,'tf' for Tensorflow
data_order ='th'
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_path ='Cat vs Dog/dataset.hdf5'  # address to where you want to save the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='w')

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)


# loop over train addresses
for i in range(len(train_addrs)):
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))

    # read an image and ...
    addr = train_addrs[i]
    img = Image.open(addr)

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image ...
    train_storage.append(img[None])

hdf5_file.close()
