'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com

Note: this repository could only be used when CUDA is available!!!
'''

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import random
import os
import sys
import argparse
import time
import dataloader
from model import MSPEC_Net,Discriminator
from Myloss import My_loss,D_loss
import numpy as np
from torchvision import transforms
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        nn.init.constant_(m.bias, 0.0)
def show_parser(args):
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']=config.gpu_device

	mspecnet = MSPEC_Net().cuda()
	mspecnet = torch.nn.DataParallel(mspecnet)

	if config.use_advloss:
		Dnet = Discriminator().cuda()
		Dnet = torch.nn.DataParallel(Dnet)

	if config.load_pretrain == True:
		mspecnet.load_state_dict(torch.load(config.pretrain_dir))
		if config.use_advloss:
			Dnet.load_state_dict(torch.load(config.D_pretrain_dir))
	else:
		config.start = [0,0]
		mspecnet.apply(weights_init)
		if config.use_advloss:
			Dnet.apply(weights_init)



	optimizer = torch.optim.Adam(mspecnet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	mspecnet.train()
	if config.use_advloss:
		D_optimizer = torch.optim.Adam(Dnet.parameters(),lr=config.D_lr,weight_decay=config.weight_decay,betas =(0.9,0.999))
		Dnet.train()
		d_loss = D_loss()
		
	for i in range(config.start[0],3):
		print(i)
		train_dataset = dataloader.dataloader(config.input_images_path,config.nomal_images_path,size=config.sizelist[i])	
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size_list[i], shuffle=True, num_workers=config.num_workers, pin_memory=True)
		mspec_loss = My_loss(size = config.sizelist[i])
		
		for epoch in range(0,config.num_epochs_list[i]):
			if i==0 and epoch+1 == 20:
				config.lr = 0.5*config.lr
				optimizer = torch.optim.Adam(mspecnet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			if ((i==1 and (epoch+1)%10 ==0) or(i==2 and (epoch+1)%5 ==0)):
				config.lr = 0.5*config.lr
				optimizer = torch.optim.Adam(mspecnet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
				if config.use_advloss:
					config.D_lr = 0.5*config.D_lr
					D_optimizer = torch.optim.Adam(Dnet.parameters(),lr=config.D_lr,weight_decay=config.weight_decay)
			if epoch < config.start[1]:
				continue
			for iteration, (lowlight_L_list,T_list) in enumerate(train_loader):
				lowlight_L_list = [data.cuda() for data in lowlight_L_list]
				T_list = [data.cuda() for data in T_list]

				if config.use_advloss:
					#train D
					D_optimizer.zero_grad()
					Y_list = mspecnet(lowlight_L_list)
					Y_list = [ Y.detach() for Y in Y_list]
					P_Y = Dnet(Y_list[-1])
					P_T = Dnet(T_list[-1])
					dloss = d_loss(P_Y,P_T)
					dloss.backward()				
					D_optimizer.step()

				#train G
				optimizer.zero_grad()
				Y_list  = mspecnet(lowlight_L_list)					
				if 	(i>0 and epoch>15)or(i>1):
					if config.use_advloss:
						P_Y = Dnet(Y_list[-1])
						rec_loss,pyr_loss,adv_loss,loss = mspec_loss(Y_list,T_list,P_Y,withoutadvloss = False)
						loss_group = {'rec_loss':rec_loss.item(),'pyr_loss':pyr_loss.item(),'adv_loss':adv_loss.item()}
					else:
						rec_loss,pyr_loss,loss = mspec_loss(Y_list,T_list,withoutadvloss = True)
						loss_group = {'rec_loss':rec_loss.item(),'pyr_loss':pyr_loss.item()}
				else:
					rec_loss,pyr_loss,loss = mspec_loss(Y_list,T_list,withoutadvloss = True)
					loss_group = {'rec_loss':rec_loss.item(),'pyr_loss':pyr_loss.item()}
				loss.backward()
				optimizer.step()
				print(loss_group)

			torchvision.utils.save_image(Y_list[-1][:,[2,1,0],:,:],'./run-out/'+config.train_mode+'/train_output4.jpg')
			torchvision.utils.save_image(Y_list[0][:,[2,1,0],:,:],'./run-out/'+config.train_mode+'/train_output1.jpg')
			torchvision.utils.save_image(Y_list[1][:,[2,1,0],:,:],'./run-out/'+config.train_mode+'/train_output2.jpg')
			torchvision.utils.save_image(Y_list[2][:,[2,1,0],:,:],'./run-out/'+config.train_mode+'/train_output3.jpg')
			torchvision.utils.save_image(T_list[-1][:,[2,1,0],:,:],'./run-out/'+config.train_mode+'/GT_example.jpg')
			torchvision.utils.save_image(lowlight_L_list[0][:,[2,1,0],:,:],'./run-out/'+config.train_mode+'/input1.jpg')
			print("Loss at epoch", epoch+1, ":", loss.item(),'Lossgroup:',loss_group)

			torch.save(mspecnet.state_dict(), config.snapshots_folder + "MSPECnet" + config.train_mode + '.pth') 
			if config.use_advloss:
				print("D_Loss at epoch", epoch+1, ":", dloss.item())
				torch.save(Dnet.state_dict(), config.snapshots_folder + "Dnet"+config.train_mode+ '.pth')		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--gpu_device', type=str, default='0')
	parser.add_argument('--train_mode', type=str, default='exp2')
	parser.add_argument('--input_images_path', type=str, default="./MultiExposure_dataset/training/Patchs/INPUT_IMAGES",help='The path of input images')
	parser.add_argument('--nomal_images_path', type=str, default="./MultiExposure_dataset/training/Patchs/GT_IMAGES",help='The path of gt images')
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--D_lr', type=float, default=1e-5)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--num_epochs_list', type=list, default= [40,20,30])
	parser.add_argument('--start', type=list, default= [0,0],help='[stage_start,epoch_start]')
	parser.add_argument('--train_batch_size_list', type=list, default=[64,32,16])
	# parser.add_argument('--train_batch_size_list', type=list, default=[256,64,32])

	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False,help='to load pretrained model')
	parser.add_argument('--sizelist',type = list, default = [128,256,512])
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/MSPECnet.pth")
	parser.add_argument('--use_advloss', action='store_true')
	parser.add_argument('--D_pretrain_dir', type=str, default= "snapshots/Dnet.pth")

	config = parser.parse_args()
	show_parser(config)
	if not os.path.exists(config.snapshots_folder):
			os.makedirs(config.snapshots_folder)

	if not os.path.exists('./run-out/'+config.train_mode):
    		os.makedirs('./run-out/'+config.train_mode)
    	

	train(config)








	
