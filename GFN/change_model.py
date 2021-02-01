


from os.path import join
import torch
from networks.GFN_4x import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#在torch 1.6版本中重新加载一下网络参数
model_name = 'F:/workplace/Python/PyTorch/deblur_lr/GFN/models/3/GFN_epoch_20.pkl'
model = Net().to(device) #实例化模型并加载到cpu货GPU中
model.load_state_dict(torch.load(model_name))  #加载模型参数，model_cp为之前训练好的模型参数（zip格式）
#重新保存网络参数，此时注意改为非zip格式
torch.save(model.state_dict(), model_name,_use_new_zipfile_serialization=False)
