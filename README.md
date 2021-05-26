# efficient-grasp
EfficientGrasp model based on EfficientPose and EfficientDet to identify robotic grasps.
## Installation

1. Install Ubuntu 18.04

2. Perform pre-installation steps of CUDA installation. Install CUDA 11.0.

3. Install libcudnn8

4. Install miniconda. Create new conda environment.
> conda create --name efficient-grasp python=3.7
> conda activate efficient-grasp
> conda install tensorflow-gpu
> conda install cv2
> pip install keras_applications

4. (Alternate) Unpack conda environment following this: https://www.anaconda.com/blog/moving-conda-environments

## Run Training

1. VSCode

    a. Go to train.py

    b. Press F5 or Ctrl-F5

2. Terminal

    > python train.py --phi 0 --batch-size 1 --lr 1e-4 --weights imagenet cornell /home/aby/Workspace/MTP/Datasets/Cornell/archive

## Run Prediction