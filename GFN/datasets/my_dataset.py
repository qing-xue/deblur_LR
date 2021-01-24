import os
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

datadir = 'F:/workplace/public_dataset/REDS/train/train_blur_bicubic/X4/000'

def find_lr_hr_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.png'):
                fullname = os.path.join(root, f)
                ref_full = fullname.replace("train_blur_bicubic/X4", "train_sharp")
                yield fullname, ref_full

class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self, imgs_lr_path):
        # xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32)  # 使用numpy读取数据
        # self.x_data = torch.from_numpy(xy[:, 0:-1])
        # self.y_data = torch.from_numpy(xy[:, [-1]])

        super(DataSet, self).__init__()     # 仍集成父类原有属性，防止def init覆盖初始化
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
        return self.x_data[index], self.sub4y_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。    
dealDataset = DealDataset(datadir)