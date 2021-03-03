import os
from os.path import join
from shutil import copyfile
import argparse
import re

parser = argparse.ArgumentParser(description="Extract 300 images")
parser.add_argument("--src", default='Results/all', help="image source path")
parser.add_argument("--dst", default='Results/300', help="image destination path")

opt = parser.parse_args()
img_path = opt.src
dst_path = opt.dst
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

for img_name in sorted(os.listdir(img_path)):
    count = int(re.split('_|\.', img_name)[0])
    sub_dir = (count - 1) / 100
    sub_name = (count - 1) % 100

    sub_dir = '%03d' % sub_dir
    sub_name = '%08d' % sub_name
    output_name = "{}_{}.png".format(sub_dir, sub_name)

    if (1 + int(sub_name)) % 10 == 0:
        copyfile(join(img_path, img_name), join(dst_path, output_name))
        print(join(img_path, img_name), '==>', join(dst_path, output_name))