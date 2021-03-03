import os
import numpy as np
import imageio
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

datadir = 'F:/workplace/public_dataset/REDS/val/val_blur_bicubic/'


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.png'):
                fullname = os.path.join(root, f)
                ref_full = fullname.replace("val_blur_bicubic/X4", "val_sharp")
                yield fullname, ref_full


def main():
    count = 0
    all_psnr = 0
    all_ssim = 0
    for bulr_file, ref_file in findAllFile(datadir):
        # print(bulr_file + '\t' + ref_file)
        im_lr = Image.open(bulr_file)
        im_sr = Image.open(ref_file)
        im_sharp = im_lr.resize((im_lr.size[0] * 4, im_lr.size[1] * 4), Image.BICUBIC)

        # PSNR and SSIM 
        ref_img = np.array(im_sr)
        res_img = np.array(im_sharp)
        psnr = peak_signal_noise_ratio(ref_img, res_img)
        ssim = structural_similarity(ref_img, res_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)

        count += 1
        all_psnr += psnr
        all_ssim += ssim
        if count % 100 == 0:
            print("%d Images, mean PSNR = %f" % (count, all_psnr / count) )    # 25.82, 25.83, 25.84
            print("%d Images, mean SSIM = %f" % (count, all_ssim / count) )    # 0.687, 0.689, 0.687


if __name__ == '__main__':
    main()