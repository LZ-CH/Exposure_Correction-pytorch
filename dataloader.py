'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''

import os
import sys
import torch
import torch.utils.data as data
from tools.decomposition import lplas_decomposition as decomposition
import numpy as np
from PIL import Image
import glob
import random
import cv2
import os
import albumentations as A
random.seed(0)

trans = A.Compose([
		A.OneOf([
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5)
		],p=0.5)
        # A.GaussNoise(p=0.2),    # 将高斯噪声应用于输入图像。
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.3),
        # 随机应用仿射变换：平移，缩放和旋转输入
        # A.RandomBrightnessContrast(p=0.5),   # 随机明亮对比度
    ])

def populate_train_list(lowlight_images_path,nomal_images_path):
	image_list_lowlight = glob.glob(lowlight_images_path + '/*')
	image_list_nomal = glob.glob(nomal_images_path + '/*')
	image_list_nomal = 5 * image_list_nomal
	image_list_lowlight.sort()
	image_list_nomal.sort()
	if len(image_list_lowlight)!=len(image_list_nomal):
		print('Data length Error')
		exit()
	return image_list_lowlight,image_list_nomal

	

class dataloader(data.Dataset):

	def __init__(self, lowlight_images_path,nomal_images_path, size ,level_num =4):

		self.image_list_lowlight, self.image_list_nomal = populate_train_list(os.path.join(lowlight_images_path,'PatchSize_'+str(size)),os.path.join(nomal_images_path,'PatchSize_'+str(size))) 
		self.size = size
		self.level_num = level_num
		self.trans = trans
		print("Total examples:", len(self.image_list_lowlight))


		

	def __getitem__(self, index):

		data_lowlight_path = self.image_list_lowlight[index]
		data_nomal_path = self.image_list_nomal[index]
		data_lowlight = cv2.imread(data_lowlight_path)
		data_nomal = cv2.imread(data_nomal_path)

		data_lowlight = data_lowlight / 255.0
		data_nomal = data_nomal / 255.0
		augment = self.trans(image = data_lowlight, mask = data_nomal)
		data_lowlight,data_nomal = augment['image'],augment['mask']
		lowlight_G_list,lowlight_L_list = decomposition(data_lowlight,self.level_num)
		lowlight_L_list = [torch.from_numpy(n).float().permute(2,0,1) for n in lowlight_L_list]

		nomal_G_list,_ = decomposition(data_nomal,self.level_num)
		nomal_G_list = [torch.from_numpy(m).float().permute(2,0,1) for m in nomal_G_list]
		return  lowlight_L_list,nomal_G_list

	def __len__(self):
		return len(self.image_list_lowlight)

