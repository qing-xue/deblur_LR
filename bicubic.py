import numpy as np
from PIL import Image

datadir = 'F:/workplace/public_dataset/REDS'

# demo image
im_sr = Image.open(datadir + '/val/val_sharp/000/00000001.png')
im_lr = im_sr.resize((im_sr.size[0] // 4, im_sr.size[1] // 4), Image.BICUBIC)

im_sr.show()
# im_lr.show()

# official blur images
im_blur = Image.open(datadir + '/val/val_blur_bicubic/X4/000/00000001.png')
im_blur.show()

# L1 loss
def L1(yhat, y):
    loss = np.mean(np.abs(y - yhat))
    return loss

print("Blur  L1 = ", (L1(np.array(im_lr), np.array(im_blur))))     # 48., 57., 67, 75., 66.

im_sharp = im_blur.resize((im_blur.size[0] * 4, im_blur.size[1] * 4), Image.BICUBIC)
im_sharp.show()
print("Sharp L1 = ", (L1(np.array(im_sharp), np.array(im_sr))))    # 114., 115., 115.

# PSNR and SSIM 
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ref_img = np.array(im_sr)
res_img = np.array(im_sharp)

psnr = peak_signal_noise_ratio(ref_img, res_img)
ssim = structural_similarity(ref_img, res_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
print("PSNR = ", (psnr))    # 25.82, 25.83, 25.84
print("SSIM = ", (ssim))    # 0.687, 0.689, 0.687