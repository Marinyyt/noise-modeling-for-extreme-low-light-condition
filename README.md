# noise-modeling-for-extreme-low-light-condition

## Background
This repository provides the code for training a denoising neural network with the synthetic data under extreme-low light environment and the corresponding testing code for evaluating the well-trained denoising network on the realistic data.  The detailed explanation about the approach has been described in the summary report:

《A Novel Noise Formation Model for Dark Scene》

In addition, it also provides some analysis codes for helping people exploring on the real noise distribution.

## Usage

### Prerequisites
Required python (version 3.6) libraries: Tensorflow (>=1.12.0) + Rawpy + imageio(2.6.1), and MATLAB R2018a.

Tested in Ubuntu 16.04/Windows 10 + Intel i9 CPU + Nvidia K80 with Cuda (>=10.1) and CuDNN (>=7.0). CPU mode should also work with minor changes but not tested.


### Required dataset
The dataset used for training and testing is Seeing In the Dark dataset (SID). SID was captured using two cameras: Sony α7SII(in the folder path "./intern/yyt/Learnin-to-see-in-the-dark/Sony/") and Fujifilm X-T2(in the folder path"./intern/yyt/Fuji/"). More specifically, for the Sony set, all the long-exposure images are recorded in "./intern/yyt/Learnin-to-see-in-the-dark/Sony/long/" and all the short-exposure images are revcorded in "./intern/yyt/Learnin-to-see-in-the-dark/Sony/short/". 

The file lists are provided(e.g., Sony_train_list.txt and Sony_test_list.txt). In the list file, each row includes a short-exposed image path, the corresponding long-exposed image path, camera ISO and F number. Note that multiple short-exposed images may correspond to the same long-exposed image.

The file name of the image contains the image information. For example, in "10019_00_0.033s.RAF", the first digit "1" means it is from the test set ("0" for training set and "2" for validation set); "0019" is the image ID; the following "00" is the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.


Another dataset is a Self-captured dark frame dataset for Sony A7R2. This dataset is mainly used for calibration based synthetic training process and for analyzing the real noise distribution. It is placed in the folder named "./Sony/SID_black level_v3". In the dataset, it contains 90 dark frames that are captured with closed lens and with no illumination environment. The file name of the image contains the image information. For example, in "8000_0.033.RAF", the number before underscore "8000" means its ISO value and the number after underscore "0.033" represents the exposure time


### Testing and Evaluation

For the testing, run "python simul_unclip_test.py" under the path "./intern/yyt/Learnin-to-see-in-the-dark/". This will load the pre-trained model from "checkpoint_dir"  and generate testing results on "result_dir" at the beginning of the code.

By default, the code takes the data in the "./intern/yyt/Learnin-to-see-in-the-dark/Sony/" folder. If you save the dataset in other folders, please change the "input_dir"(the directory recorded short-exposure images) and "gt_dir"(the directory recorded long-exposure images) at the beginning of the code.

For the evaluation part, run "python evaluation.py" in the "result_dir" that has been set in the testing code before. This will compute the mean PSNR and SSIM score for the whole testing images and print it on the screen. 


### Training new models
To train the model, run "python train_xxx.py" under the path "./intern/yyt/Learnin-to-see-in-the-dark/". The model will be save in "result_Sony" folder by default.

In summary, I provide some training codes that I think are more important:
* "train_clip.py" is the initial version for training with synthetic data, the synthetic data is generated with heteroskedastic Gaussian noise model.
* "train_bl_pert.py" is the version of adding perturbation operation. The hyper-parameter "a" that controls the magnitude of perturbation is at the begining of the code.
* "train_bl_cali_patch.py" is the version of correcting black level by using dark frame dataset.
* "train_bl_cali_full.py" is an extented version of "train_bl_cali_patch.py", training with the full-length image.
* "train_gamma.py" is the version of adding row noise which follows gamma distribution with a shape parameter extracted from a uniform distribution and an increasingly scale parameter.
* "train_gamma_sequential.py" is the version of adding row noise which follows gamma distribution with shape 1 and an increasingly scale parameter.
* "train_gamma_calib.py" is the version of adding row noise extracted from dark frame data set.

By default, the code takes the data in the "./dataset/Sony/" folder. If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.



### Analysis Code
"sonytest.m" is the MATlAB code for processing a raw image in the path "filename" into a displayable sRGB image. "wbmask" and "apply_cmatrix" are the function code used in "sonytest.m".


