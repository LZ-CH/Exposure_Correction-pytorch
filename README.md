# Exposure_Correction-pytorch
    此项目为CVPR2021关于图像光照修正领域的论文[Learning Multi-Scale Photo Exposure Correction.](https://arxiv.org/pdf/2003.11596.pdf)的非官方pytorch复现代码；前些
    
日子里阅读到了这篇十分有趣的论文[Learning Multi-Scale Photo Exposure Correction.](https://arxiv.org/pdf/2003.11596.pdf)，本想基于其源代码进行修改，但是该论文的官方代

码是MATLAB实现的，对于一个pytorch惯用者来说，难免有点不顺手，因此本人花了一些时间对该论文采用pytorch框架进行了简单复现；在不采用原MATLAB代码中使用的 
    
    Bilateral Guided Upsampling (bgu)上采样方式而直接采用简单的上采样方式对预测结果进行处理，复现结果为psnr: 19.756，ssim: 0.749；若采用后，复现结果为psnr: 20.313 

SSIM: 0.863；(原论文在相同方式下:psnr: 20.205，ssim: 0.769)
    

## Folder structure
该部分的项目文件夹应按以下结构放置:
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
│   ├── MSPECnet_woadv.pth #  预训练模型
```
## Requirements
所需环境配置
1. Python  3.8.0
2. Pytorch 1.9.1
3. numpy   1.21.0

## prepare data
1. 首先从[官方github仓库](https://github.com/mahmoudnafifi/Exposure_Correction)中下载[Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj)|[Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr)|[Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxikex-q7vydzkp) 等数据集
2. 将数据集按照文件夹结果放置在该项目的根目录下
3. 运行以下代码对数据进行切分预处理,之后会在./MultiExposure_dataset/training目录下新生成一个Patchs文件夹
```
python ./tools/creat_patch.py
```
## train
1. 运行以下命令进行无对抗损失训练:
```
python mspec_train.py
```

2. 如果想将对抗损失加入到训练当中，则运行:
```
python mspec_train.py --use_advloss
```

## test
1. 可以直接下载本人在无对抗损失下训练好的checkpoint[baidu clound password: 1234](https://pan.baidu.com/s/1GlXrhQfdasCPStcPp5ahyQ)
2. 也可以自己训练得到checkpoint
3. 然后运行以下命令进行测试验证:
```
python mspec_test.py
```
## Contact information
E-mail: 2443976970@qq.com
