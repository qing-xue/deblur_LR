import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os

class _ResBLockDB(nn.Module):
    """ResBlocks in Deblur module

    """
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))    # BN? 
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _ResBlockSR(nn.Module):
    """ResBlocks in SR module

    """

    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlockSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out


class _DeblurringMoudle(nn.Module):
    def __init__(self):
        super(_DeblurringMoudle, self).__init__()
        self.conv1     = nn.Conv2d(3, 64, (7, 7), 1, padding=3)
        self.relu      = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock1 = self._makelayers(64, 64, 6)    # 添加多个残差块
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), 2, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock2 = self._makelayers(128, 128, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), 2, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock3 = self._makelayers(256, 256, 6)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (7, 7), 1, padding=3)
        )
        self.convout = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, (3, 3), 1, 1)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1   = self.relu(self.conv1(x))      # 64,strided=1
        res1   = self.resBlock1(con1)
        res1   = torch.add(res1, con1)         # 第一个跳跃连接
        con2   = self.conv2(res1)              # 128，strided=2
        res2   = self.resBlock2(con2)
        res2   = torch.add(res2, con2)         # 第二个跳跃连接
        con3   = self.conv3(res2)              # 256
        res3   = self.resBlock3(con3)
        res3   = torch.add(res3, con3)         # 第三个跳跃连接
        decon1 = self.deconv1(res3)            # 反卷积，128
        deblur_feature = self.deconv2(decon1)  # 内部在激活层内加了卷积
        deblur_out = self.convout(torch.add(deblur_feature, con1))  # 全局跳跃连接，再接convout
        return deblur_feature, deblur_out      # 返回deblur特征图和低分辨率的deblur输出


class _SRMoudle(nn.Module):
    """Super-resolution feature extraction module.

    We use eight ResBlocks [20] to extract high-dimensional features for image super-resolution.
    """
    def __init__(self):
        super(_SRMoudle, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), 1, padding=3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock = self._makelayers(64, 64, 8, 1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBlockSR(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        res1 = self.resBlock(con1)
        con2 = self.conv2(res1)
        sr_feature = torch.add(con2, con1)
        return sr_feature


class _GateMoudle(nn.Module):
    def __init__(self):
        super(_GateMoudle, self).__init__()

        # 64 + 64 + 3 = 131
        self.conv1 = nn.Conv2d(131,  64, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.conv2(con1)
        return scoremap


class _ReconstructMoudle(nn.Module):
    """Reconstruction module.

    The fused features φfusion from the gate module are fed into 8 ResBlocks 
    and 2 pixel-shuffling layers [34] to enlarge the spatial resolution by 4×. 
    We then use 2 final convolutional layers to reconstruct an HR output image H?.
    """

    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
        self.resBlock = self._makelayers(64, 64, 8)
        self.conv1 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64, 3, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1))
        con2 = self.conv2(pixelshuffle1)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))
        con3 = self.relu3(self.conv3(pixelshuffle2))
        sr_deblur = self.conv4(con3)
        return sr_deblur


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurMoudle      = self._make_net(_DeblurringMoudle)
        self.srMoudle          = self._make_net(_SRMoudle)
        self.geteMoudle        = self._make_net(_GateMoudle)
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, x, gated, isTest):
        if isTest == True:
            origin_size = x.size()
            input_size  = (math.ceil(origin_size[2]/4)*4, math.ceil(origin_size[3]/4)*4)
            out_size    = (origin_size[2]*4, origin_size[3]*4)
            x           = nn.functional.upsample(x, size=input_size, mode='bilinear')  # 有无上采样？

        deblur_feature, deblur_out = self.deblurMoudle(x)
        sr_feature = self.srMoudle(x)
        if gated == True:
            # torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。
            scoremap = self.geteMoudle(torch.cat((deblur_feature, x, sr_feature), 1))
        else:
            scoremap = torch.cuda.FloatTensor().resize_(sr_feature.shape).zero_()+1
        repair_feature = torch.mul(scoremap, deblur_feature)
        fusion_feature = torch.add(sr_feature, repair_feature)
        recon_out = self.reconstructMoudle(fusion_feature)

        if isTest == True:
            recon_out = nn.functional.upsample(recon_out, size=out_size, mode='bilinear')

        return deblur_out, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)




