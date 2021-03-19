# -*- coding: utf-8 -*-
import torch

# 从 losses.py 导入变量或函数都不行！！！！
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used:", device)
