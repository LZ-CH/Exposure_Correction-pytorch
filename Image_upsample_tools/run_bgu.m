%author LZ-CH
%Email 2443976970@qq.com
%using Bilateral Guided Upsampling(bgu) to upsample images.
clear;clc;
exts = {'.jpg','.jpeg','.png','.tif','.bmp'};
input_dir = '.\DataSet\MultiExposure_dataset\testing\INPUT_IMAGES\';	%Input iamges dir (with the same size to GT).
gt_dir = '.\DataSet\MultiExposure_dataset\testing\expert_c_testing_set\';	% ground truth dir (GT).
test_dir = '.\DataSet\MultiExposure_dataset\testing\test_out\';% model output dir (small size).
final_output_dir = '.\DataSet\MultiExposure_dataset\testing\test_out_BGU'; %bgu output dir
gt_images = {};
test_images = {};
input_images ={};
inSz = 512; %the size of model output images
tempSz = 200; %downsample size. use a smaller size to speed up Bilateral Guided Upsampling.
if exist(final_output_dir,'dir') == 0
    mkdir(final_output_dir);
end
for i = 1 : length(exts)
    temp_files = dir(fullfile(input_dir,['*' exts{i}]));
    input_images = [input_images; {temp_files(:).name}'];
end
for i = 1 : length(exts)
    temp_files = dir(fullfile(gt_dir,['*' exts{i}]));
    gt_images = [gt_images; {temp_files(:).name}'];
end
for i = 1 : length(exts)
    temp_files = dir(fullfile(test_dir,['*' exts{i}]));
    test_images = [test_images; {temp_files(:).name}'];
end
gt_images = [gt_images;gt_images;gt_images;gt_images;gt_images];
input_images = sort(input_images);
gt_images = sort(gt_images);
test_images = sort(test_images);
length(test_images)
for i = 1 : length(test_images)
    input_imageName = input_images{i};
    test_imageName = test_images{i};
    
    I = im2double(imread(fullfile(input_dir,input_imageName)));
    I_ = I;
    output = im2double(imread(fullfile(test_dir,test_imageName)));
    sz = size(I);
    if (max(sz) > inSz) == 1
        disp('Upsampling...');
        tic
        output_s = double(imresize(output,[tempSz,tempSz]));
        I = imresize(I,inSz/max(sz));
        I = double(imresize(I,[tempSz,tempSz]));
        results = computeBGU(I, rgb2luminance(I), output_s, [], ...
            I_, rgb2luminance(I_));
        output = results.result_fs;
        fprintf('Upsampling time: %f seconds.\n',toc);
    else
        output = imresize(output,[sz(1) sz(2)]);
    end
    imwrite(output,fullfile(final_output_dir,test_imageName));

end
