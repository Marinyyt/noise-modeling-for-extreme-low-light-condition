# noise-modeling-for-extreme-low-light-condition

## Background
This repository provides the code for training a denoised neural network with the synthetic data under extreme-low light environment and the corresponding testing code for evaluating the well-trained network on the realistic data.  The detailed illustration is described in the summary report:

《A Novel Noise Formation Model for Dark Scene》

In addition, it also provides some analysis codes for exploring on the real noise distribution.

## Usage

### Prerequisites
Required python (version 2.7) libraries: Tensorflow (>=1.1) + Scipy + Numpy + Rawpy.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes but not tested.


### Required dataset
Seeing In the Dark dataset (SID). More specifically, it is recommended to use the sony sets for training and testing our code. 

The file lists are provided(e.g., Sony_train_list.txt). In the list file, each row includes a short-exposed image path, the corresponding long-exposed image path, camera ISO and F number. Note that multiple short-exposed images may correspond to the same long-exposed image.

The file name of the image contains the image information. For example, in "10019_00_0.033s.RAF", the first digit "1" means it is from the test set ("0" for training set and "2" for validation set); "0019" is the image ID; the following "00" is the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.



Another dataset is a Self-captured dark frame dataset for Sony A7R2. This dataset is mainly used for calibration based synthetic training process and for analyzing the real noise distribution. It is placed in the folder named "SID_black level_v3". In the dataset, it contains 90 dark frames that are captured with closed lens and with no illumination environment. The file name of the image contains the image information. For example, in "8000_0.033.RAF", the number before underscore "8000" means its ISO value and the number after underscore "0.033" represents the exposure time


### Testing and Evaluation

For the testing part, run "python simul_unnclip_test.py". This will load the pre-trained model from "checkpoint_dir" at the beginning of the code and generate testing results on "result_dir".

By default, the code takes the data in the "./Sony/" folder. If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.

For the evaluation part, run "python evaluation.py" in the "result_dir" that is set in the testing code before. This will compute the mean PSNR and SSIM score for the whole testing images and finally print its computation result on the screen. 


### Training new models
To train the Sony model, run "python train_Sony.py". The result and model will be save in "result_Sony" folder by default.
To train the Fuji model, run "python train_Fuji.py". The result and model will be save in "result_Fuji" folder by default.
By default, the code takes the data in the "./dataset/Sony/" folder and "./dataset/Fuji/". If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.
