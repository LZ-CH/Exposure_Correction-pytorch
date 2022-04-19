'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''


from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
from glob import glob
import cv2

def calculate_psnr_ssim(img_dir,gtimg_dir):
    imgpaths = glob(img_dir+'/*')
    gtpaths = glob(gtimg_dir+'/*')*5
    imgpaths.sort()
    gtpaths.sort()
    psnr_list = []
    ssim_list = []
    print('Image num:',len(imgpaths))
    for n in range(len(imgpaths)):
        img = cv2.imread(imgpaths[n])
        gtimg = cv2.imread(gtpaths[n])
        assert img.shape == gtimg.shape
        psnr = calculate_psnr(img,gtimg)
        ssim = calculate_ssim(img, gtimg,multichannel = True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        # if psnr<10:
        #     print('PSNR:',psnr,imgpaths[n],'------>',gtpaths[n])
    print('Min_PSNR:',np.min(psnr_list),'Min_ssim:',np.min(ssim_list))
    print('PSNR:',np.mean(psnr_list),'SSIM:',np.mean(ssim_list))
    return np.mean(psnr_list),np.mean(ssim_list)
if __name__ == '__main__':
    img_dir = './MultiExposure_dataset/testing/eval_output'
    gtimg_dir = './MultiExposure_dataset/testing/expert_c_testing_set'
    psnr,ssim = calculate_psnr_ssim(img_dir,gtimg_dir)
