import os
from os.path import join
from shutil import copyfile

img_path = 'Results/Results'
dst_path = 'Results'

for img_name in sorted(os.listdir(img_path)):
    count = int(img_name.split('_')[0])
    sub_dir = (count - 1) / 100
    sub_name = (count - 1) % 100

    sub_dir = '%03d' % sub_dir
    sub_name = '%08d' % sub_name
    output_name = "{}_{}.png".format(sub_dir, sub_name)

    if (1 + int(sub_name)) % 10 == 0:
        copyfile(join(img_path, img_name), join(dst_path, output_name))