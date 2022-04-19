# Exposure_Correction-pytorch
English | [简体中文](README.md)

This project is the unofficial pytorch reproduction code of the CVPR2021 paper on the field of image illumination correction [Learning Multi-Scale Photo Exposure Correction.](https://arxiv.org/pdf/2003.11596.pdf);
    
I read this very interesting paper [Learning Multi-Scale Photo Exposure Correction.](https://arxiv.org/pdf/2003.11596.pdf) a few days ago. I wanted to modify it based on its source code, but the paper The official code of MATLAB is implemented by MATLAB. For a pytorch user, it is inevitable that it is a bit awkward. Therefore, I spent some time using the pytorch framework to reproduce this paper. The Bilateral Guided used in the original MATLAB code is not used. Upsampling (bgu) upsampling method directly uses simple upsampling method to process the prediction result, the recurrence result is psnr: 19.756, ssim: 0.749; if adopted, the recurrence result is psnr: 20.313, SSIM: 0.863; ( Original paper in the same way: psnr: 20.205, ssim: 0.769)
    

## Folder structure
The project folder for this section should be placed in the following structure:
```
Exposure_Correction-pytorch
├── MultiExposure_dataset
│   ├── testing
│   ├── training
│   └── validation
├── log
├── run-out
├── tools
├── snapshots
│   ├── MSPECnet_woadv.pth # pretrained model
```
## Requirements

1. Python  3.8.0
2. Pytorch 1.9.1
3. numpy   1.21.0

## prepare data
1. First download [Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj)|[ Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr)|[Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxiex-q7vydzkp)from the [official github repository](https://github.com/mahmoudnafifi/Exposure_Correction)
2. Place the dataset in the root directory of the project according to the folder result
3. Run the following code to preprocess the data, and then a new Patchs folder will be generated in the ./MultiExposure_dataset/training directory
```
python ./tools/creat_patch.py
```
## train
1. Run the following command for training without adversarial loss:
```
python mspec_train.py
```

2. If you want to add an adversarial loss to training, run:
```
python mspec_train.py --use_advloss
```

## test
1. You can directly download the checkpoint that I trained without confrontation loss: [baidu clound password: 1234](https://pan.baidu.com/s/1GlXrhQfdasCPStcPp5ahyQ)
2. You can also train the model yourself
3. Then run the following command for test verification:
```
python mspec_test.py
```
## Contact information
E-mail: 2443976970@qq.com
