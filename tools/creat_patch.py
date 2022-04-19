'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''

import cv2
import numpy as np
from glob import glob
import random
import os
random.seed(0)
def is_discard(rawimg):
    img = rawimg.copy()
    if img.dtype=='uint8':
        img = img/255
    if np.mean(img)<0.06 or np.mean(img)>0.98:
        return True
    grad_X=cv2.Sobel(img,-1,1,0)
    grad_Y=cv2.Sobel(img,-1,0,1)

    grad = grad_X + grad_Y
    
    if np.mean(grad)<0.01:
        return True
    else:
        return False

if __name__ =='__main__':
    Pnum_per_img = 20
    input_image_path = './MultiExposure_dataset/training/INPUT_IMAGES'
    gt_image_path = './MultiExposure_dataset/training/GT_IMAGES'
    save_dir = './MultiExposure_dataset/training/Patchs'
    patch_input = os.path.join(save_dir,'INPUT_IMAGES')
    patch_gt = os.path.join(save_dir,'GT_IMAGES')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(patch_input):
        os.makedirs(patch_input)
    if not os.path.exists(patch_gt):
        os.makedirs(patch_gt)
    input_list = glob(input_image_path+'/*')
    gt_list = glob(gt_image_path+'/*')
    input_list.sort()
    gt_list.sort()
    patch_size_list = [128,256,512]
    getsum = [0,0,0]

    for k ,patch_size in enumerate(patch_size_list):
        size_input_dir = os.path.join(patch_input,'PatchSize_'+str(patch_size))
        size_gt_dir = os.path.join(patch_gt,'PatchSize_'+str(patch_size))                
                
        if not os.path.exists(size_input_dir):
            os.mkdir(size_input_dir)
        if not os.path.exists(size_gt_dir):
            os.mkdir(size_gt_dir)

        for n, gtpath in enumerate(gt_list):
            discard_count = 0
            get_count = 0
            max_discard = 200
            gtimg = cv2.imread(gtpath)
            for m in range(5):
                eachimg = cv2.imread(input_list[5*n+m])
                if eachimg.shape != gtimg.shape:
                    discard_count = max_discard
                    print('Discard all.')
                    break

            if (gtimg.shape[0]<patch_size or gtimg.shape[1]<patch_size):
                continue
            while True:
                if discard_count==max_discard :
                    break
                r1 = random.randint(0, gtimg.shape[0]-patch_size)
                r2 = random.randint(0, gtimg.shape[1]-patch_size)
                gtpatch = gtimg[r1:r1+patch_size,r2:r2+patch_size,:]
                assert gtpatch.shape[0]==patch_size and gtpatch.shape[1]==patch_size
                if is_discard(gtpatch):
                    discard_count += 1
                    continue
                get_count +=1
                gtbasename = os.path.basename(gtpath) 
                for j in range(5):
                    inputpath = input_list[5*n+j]
                    input_img = cv2.imread(inputpath)
                    inputbasename = os.path.basename(inputpath)
                    inserpos = inputbasename.rfind('_')
                    input_patch = input_img[r1:r1+patch_size,r2:r2+patch_size,:]
                    if input_patch.shape[0]!=patch_size or input_patch.shape[1]!=patch_size:
                        print('Error!',input_patch.shape,gtpatch.shape,r1,r2,input_img.shape,gtimg.shape)
                        print(inputpath,gtpath)
                        exit()
                    cv2.imwrite(os.path.join(size_input_dir,inputbasename[:inserpos]+'_'+str(get_count).zfill(2)+inputbasename[inserpos:]),input_patch)
                cv2.imwrite(os.path.join(size_gt_dir,os.path.splitext(gtbasename)[0]+'_'+str(get_count).zfill(2)+os.path.splitext(gtbasename)[1]),gtpatch)
                if  get_count == Pnum_per_img:
                    break
            getsum[k] = getsum[k]+get_count
            print(get_count,getsum)
            # exit()

